import { base } from "@account-kit/infra";
import Anthropic from "@anthropic-ai/sdk";
import dotenv from "dotenv";
import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AcpAgent,
  AssetToken,
  PrivyAlchemyEvmProviderAdapter,
  type AcpAgentOffering,
  type JobRoomEntry,
  type JobSession,
} from "@virtuals-protocol/acp-node-v2";

dotenv.config({ quiet: true });

// ---------------------------------------------------------------------------
// Digest: a minimal ACP provider (seller) agent.
//
// Registered offering:
//   name         textDigest
//   requirement  { text: string }     up to ~2000 words
//   deliverable  { summary: string }  2-3 sentences
//
// Protocol behaviour here is plain deterministic code: the same job always
// produces the same accept/reject decision and the same budget. A language
// model is used only to write the summary prose, never to decide whether or
// how to transact. That mirrors the determinism contract Priors holds itself
// to, applied one layer out to its counterparty.
//
// Required env (see .env.example):
//   SELLER_WALLET_ADDRESS, SELLER_WALLET_ID, SELLER_SIGNER_PRIVATE_KEY
// Optional:
//   ANTHROPIC_API_KEY  : without it, Digest falls back to extractive
//                        summarisation so a demo never dies on a missing key.
// ---------------------------------------------------------------------------

const OFFERING_NAME = "textDigest";

// Coordination. Digest answers the requests Priors writes to the shared store,
// which is what makes the store the coordination surface rather than a
// noticeboard: it responds to a specific reference, it does not announce
// itself. The memory client is Python, so this shells out to the same scripts
// the buyer uses; a second implementation of the store in another language is
// the one thing that could put the two agents out of step.
const RESPOND_EVERY_MS = Number(process.env.RESPOND_EVERY_MS ?? 10_000);

// Controlled fault injection. Digest declares itself available and then
// declines to act on that claim, so Priors can observe a broken promise. It is
// a count of upcoming jobs, set by hand and visible in the environment, so a
// judge re-running the demo gets the same behaviour. Nothing random decides it.
let faultsRemaining = Number(process.env.DIGEST_FAULT_NEXT ?? 0);
// Which jobs have already been chosen for the fault. The requirement handler
// can fire more than once for a job - a replayed entry, or a rehydrated
// session - and the usual `status === "open"` guard does not stop it here,
// because declining is precisely what leaves the status open. Counting down a
// global tally would therefore fault the first pass and accept the second,
// which is exactly what happened on job 76935.
const faultedJobs = new Set<string>();
const MAX_WORDS = 2000;
// Haiku 4.5 is the cheapest tier ($1/$5 per 1M in/out) and ample for a 2-3
// sentence summary, roughly $0.003 per job at the 2000-word ceiling. It takes
// neither `thinking` nor `effort`, so this stays a plain request.
const SUMMARY_MODEL = "claude-haiku-4-5";

const shortAddr = (a: string): string =>
  !a || !a.startsWith("0x") || a.length < 12
    ? a
    : `${a.slice(0, 6)}…${a.slice(-4)}`;

const log = {
  info: (m: string) => console.log(`[digest] ${m}`),
  job: (id: string | number, m: string) =>
    console.log(`[digest] [job ${id}] ${m}`),
  warn: (m: string) => console.warn(`[digest] [warn] ${m}`),
  error: (m: string, e?: unknown) =>
    console.error(`[digest] [error] ${m}`, e ?? ""),
};

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env var: ${name}`);
  return v;
}

const wordCount = (s: string): number => (s.trim().match(/\S+/g) ?? []).length;

const anthropic = process.env.ANTHROPIC_API_KEY ? new Anthropic() : null;

/** First few sentences, the deterministic fallback when no model is available. */
function extractiveSummary(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const sentences = flat.match(/[^.!?]+[.!?]+/g) ?? [];
  const picked = sentences.slice(0, 3).join(" ").trim();
  return picked || flat.slice(0, 300);
}

async function summarize(
  text: string
): Promise<{ summary: string; via: string }> {
  if (anthropic) {
    try {
      const msg = await anthropic.messages.create({
        model: SUMMARY_MODEL,
        max_tokens: 300,
        system:
          "Summarise the user's text in 2-3 sentences. Reply with the summary only: no preamble, no bullet points, no restating the request.",
        messages: [{ role: "user", content: text }],
      });
      const out = msg.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("")
        .trim();
      if (out) return { summary: out, via: SUMMARY_MODEL };
      log.warn("model returned empty text; using extractive fallback");
    } catch (err) {
      log.warn(`model call failed, using extractive fallback: ${err}`);
    }
  }
  return { summary: extractiveSummary(text), via: "extractive-fallback" };
}

/**
 * Recover the buyer's requirement from the job transcript.
 *
 * `session.entries` is hydrated with the job's history when the SDK loads a
 * session, so this still works if Digest restarts between the requirement
 * arriving and the job being funded.
 */
function findRequirementText(session: JobSession): string | undefined {
  for (const e of session.entries) {
    if (e.kind !== "message" || e.contentType !== "requirement") continue;
    try {
      const parsed = JSON.parse(e.content) as unknown;
      const text = (parsed as { text?: unknown })?.text;
      if (typeof text === "string" && text.trim()) return text;
    } catch {
      // A malformed requirement was already rejected at the requirement stage.
    }
  }
  return undefined;
}

type Validation =
  | { ok: true; text: string }
  | { ok: false; tag: string; detail: string };

function validateRequirement(raw: unknown): Validation {
  const text = (raw as { text?: unknown })?.text;
  if (typeof text !== "string") {
    return {
      ok: false,
      tag: "missing text",
      detail: 'Requirement must include a "text" field of type string.',
    };
  }
  if (!text.trim()) {
    return {
      ok: false,
      tag: "empty text",
      detail: 'The "text" field is empty.',
    };
  }

  const words = wordCount(text);
  if (words > MAX_WORDS) {
    return {
      ok: false,
      tag: "text too long",
      detail: `Input is ${words} words; this offering accepts up to ${MAX_WORDS}.`,
    };
  }

  return { ok: true, text };
}

async function main(): Promise<void> {
  const seller = await AcpAgent.create({
    evmProvider: await PrivyAlchemyEvmProviderAdapter.create({
      walletAddress: requireEnv("SELLER_WALLET_ADDRESS") as `0x${string}`,
      walletId: requireEnv("SELLER_WALLET_ID"),
      signerPrivateKey: requireEnv("SELLER_SIGNER_PRIVATE_KEY"),
      chains: [base],
    }),
  });

  const sellerAddress = (await seller.getAddress()).toLowerCase();
  log.info(`address: ${sellerAddress}`);
  log.info(
    anthropic
      ? `summariser: ${SUMMARY_MODEL}`
      : "summariser: extractive fallback (ANTHROPIC_API_KEY not set)"
  );

  // Price comes from the registry, never hardcoded, so editing the offering
  // on app.virtuals.io is enough to change what Digest quotes.
  let offeringsByName = new Map<string, AcpAgentOffering>();
  try {
    const me = await seller.getAgentByWalletAddress(sellerAddress);
    offeringsByName = new Map(
      (me?.offerings ?? []).map((o) => [o.name, o] as const)
    );
    log.info(`loaded ${offeringsByName.size} offering(s):`);
    for (const o of offeringsByName.values()) {
      log.info(`  - ${o.name}: ${o.priceValue} USDC (sla=${o.slaMinutes}min)`);
    }
  } catch (err) {
    log.error("failed to load registry offerings", err);
    throw err;
  }

  if (!offeringsByName.has(OFFERING_NAME)) {
    throw new Error(
      `Offering "${OFFERING_NAME}" is not registered for this wallet. ` +
        "Add it under ACP -> Offerings -> Jobs Offered before starting Digest."
    );
  }

  seller.on("entry", async (session: JobSession, entry: JobRoomEntry) => {
    if (entry.kind === "system") {
      switch (entry.event.type) {
        case "job.created":
          log.job(
            session.jobId,
            `new job from buyer ${shortAddr(entry.event.client)}`
          );
          break;

        case "job.funded": {
          log.job(session.jobId, "funded, delivering");
          try {
            const text = findRequirementText(session);
            if (!text) {
              // Should be unreachable: a job only reaches funding after we
              // accepted its requirement and set a budget.
              log.error(`no requirement text on funded job ${session.jobId}`);
              await session.sendMessage(
                "Could not recover the original request text; unable to deliver."
              );
              await session.reject("requirement lost");
              return;
            }

            const { summary, via } = await summarize(text);
            log.job(
              session.jobId,
              `summarised ${wordCount(text)} words via ${via}`
            );
            await session.submit(JSON.stringify({ summary }));
            log.job(session.jobId, "submitted deliverable");
          } catch (err) {
            log.error(`delivery failed on job ${session.jobId}`, err);
          }
          break;
        }

        case "job.completed":
          log.job(session.jobId, "completed");
          break;

        case "job.rejected":
          log.job(
            session.jobId,
            `rejected by ${shortAddr(entry.event.rejector)}: ${entry.event.reason}`
          );
          break;

        case "job.expired":
          log.job(session.jobId, "expired");
          break;
      }
    }

    // The buyer's first message carries the structured requirement. The
    // `status === "open"` guard makes replayed entries idempotent: once a
    // budget is set the status advances and this block stops firing.
    if (
      entry.kind === "message" &&
      entry.contentType === "requirement" &&
      session.status === "open"
    ) {
      const rejectWithDetail = async (
        tag: string,
        detail: string
      ): Promise<void> => {
        log.job(session.jobId, `rejecting: ${detail} (tag: "${tag}")`);
        await session.sendMessage(detail);
        await session.reject(tag);
      };

      // Routing uses the on-chain offering name, not the message envelope.
      const offeringName = session.job?.description;
      if (!offeringName) {
        await rejectWithDetail(
          "missing offering name",
          "Job description is empty; cannot identify the offering."
        );
        return;
      }

      const offering = offeringsByName.get(offeringName);
      if (!offering || offeringName !== OFFERING_NAME) {
        await rejectWithDetail(
          "unsupported offering",
          `This agent only serves "${OFFERING_NAME}"; got "${offeringName}".`
        );
        return;
      }

      let requirementData: unknown;
      try {
        requirementData = JSON.parse(entry.content);
      } catch (err) {
        await rejectWithDetail(
          "unparseable requirement",
          `Could not parse the requirement payload: ${err}`
        );
        return;
      }

      const check = validateRequirement(requirementData);
      if (!check.ok) {
        await rejectWithDetail(check.tag, check.detail);
        return;
      }

      const jobKey = String(session.jobId);
      if (faultedJobs.has(jobKey) || faultsRemaining > 0) {
        if (!faultedJobs.has(jobKey)) {
          faultedJobs.add(jobKey);
          faultsRemaining -= 1;
          log.warn(
            `[job ${jobKey}] FAULT INJECTED: declared available, declining ` +
              `to accept. ${faultsRemaining} further job(s) will be faulted.`
          );
        }
        // No setBudget, no rejection. The job is simply left to expire, which
        // is what a broken availability claim looks like on chain: created,
        // never answered. Rejecting instead would be a different signal.
        return;
      }

      log.job(
        session.jobId,
        `accepted "${offeringName}", ${wordCount(check.text)} words`
      );

      try {
        await session.setBudget(
          AssetToken.usdc(offering.priceValue, session.chainId)
        );
        log.job(session.jobId, `set budget to ${offering.priceValue} USDC`);
      } catch (err) {
        log.error(`setBudget failed on job ${session.jobId}`, err);
      }
    }
  });

  await seller.start();
  // Answer whatever Priors has asked, on a timer. execFile rather than
  // spawnSync: a synchronous call here would stall the ACP event loop while it
  // waits on a chain read.
  const respond = () => {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const script = path.resolve(here, "../../scripts/respond.py");
    execFile(
      process.env.PYTHON ?? "python",
      [script, "--agent", sellerAddress, "--status", "available"],
      (err, stdout) => {
        if (err) {
          log.warn(`respond.py failed: ${String(err).slice(0, 120)}`);
          return;
        }
        try {
          const out = JSON.parse(stdout) as { answered: number };
          if (out.answered > 0) {
            log.info(`declared availability on ${out.answered} request(s)`);
          }
        } catch {
          /* a malformed reply is not worth taking the agent down for */
        }
      }
    );
  };
  respond();
  const responder = setInterval(respond, RESPOND_EVERY_MS);
  responder.unref();

  if (faultsRemaining > 0) {
    log.warn(
      `FAULT INJECTION ARMED: will decline the next ${faultsRemaining} job(s) ` +
        "after declaring availability"
    );
  }

  log.info("ready, listening for jobs");

  const shutdown = async (signal: NodeJS.Signals) => {
    log.info(`received ${signal}, shutting down`);
    await seller.stop();
    process.exit(0);
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}

main().catch((err) => {
  log.error("fatal", err);
  process.exit(1);
});
