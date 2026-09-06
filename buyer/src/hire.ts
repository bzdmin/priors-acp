import { base } from "@account-kit/infra";
import dotenv from "dotenv";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AcpAgent,
  PrivyAlchemyEvmProviderAdapter,
  type JobRoomEntry,
  type JobSession,
} from "@virtuals-protocol/acp-node-v2";

dotenv.config({ quiet: true });

// ---------------------------------------------------------------------------
// Priors' buyer runtime: decide first, then transact.
//
// The decision is not made here. This shells out to `scripts/decide.py`, which
// runs the same `predict` the backtest and the live tail call, and refuses to
// spend anything unless that returns HIRE. Reimplementing the predictor in
// TypeScript would give two copies of the arithmetic that could drift apart,
// and then the live hire would no longer be evidence about the model that was
// actually evaluated.
//
// Evaluation mode: `evaluatorAddress` is deliberately omitted, which puts the
// job in skip-evaluation mode. Priors does NOT appoint itself evaluator. Its
// own research found that on 59% of completed ACP jobs the evaluator is the
// client who created the job, and called that out as a mechanism not being
// used as intended - so doing it here would be the thing it criticised.
//
// The outcome does not need recording from this process. Once the job is
// on-chain it appears in ACP's logs like any other, and `scripts/live_tail.py`
// decodes it, decides on it, and lets the resolver write the episode. The loop
// closes through the same path as every other job.
//
// Required env (see .env.example):
//   PRIORS_WALLET_ADDRESS, PRIORS_WALLET_ID, PRIORS_SIGNER_PRIVATE_KEY
// ---------------------------------------------------------------------------

const OFFERING_NAME = "textDigest";
const DEFAULT_PROVIDER = "0x5043147b8b666ac070e01ff659e1fbbbc2462bc7";
const CHAIN_ID = base.id;

/** Small enough that a stuck job locks almost nothing until it expires. */
const DEFAULT_AMOUNT_USDC = 0.02;

const SAMPLE_TEXT = `The Agent Commerce Protocol lets autonomous agents hire
one another on-chain. A client creates a job, a provider accepts and sets a
budget, the client funds escrow, the provider delivers, and payment is
released. An optional evaluator can gate payment on the quality of the
deliverable. In practice the evaluator role is rarely filled by an
independent party: across 14,644 completed jobs, a genuine third party
appeared 18 times, while the client who created the job acted as its own
evaluator on 8,659 of them. Funding is the other filter - only about a
quarter of created jobs are ever funded at all.`;

const shortAddr = (a: string): string =>
  !a || !a.startsWith("0x") || a.length < 12
    ? a
    : `${a.slice(0, 6)}…${a.slice(-4)}`;

const log = {
  info: (m: string) => console.log(`[priors] ${m}`),
  job: (id: string | number, m: string) =>
    console.log(`[priors] [job ${id}] ${m}`),
  warn: (m: string) => console.warn(`[priors] [warn] ${m}`),
  error: (m: string, e?: unknown) =>
    console.error(`[priors] [error] ${m}`, e ?? ""),
};

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env var: ${name}`);
  return v;
}

type Decision = {
  decision: string;
  hire: boolean;
  with_memory: { p: number; decision: string; rules_applied: RuleApplied[] };
  chain_only: { p: number; decision: string };
  memory_changed_probability: boolean;
  memory_changed_decision: boolean;
  provider_record: { funded: number; completed: number };
  store: string | null;
};

type RuleApplied = {
  rule_id: string;
  delta_logit: number;
  weight: number;
  why: string;
};

/** Ask Priors. Throws rather than guessing if the predictor cannot be run. */
function askPriors(provider: string, amountUsdc: number): Decision {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const script = path.resolve(here, "../../scripts/decide.py");
  const python = process.env.PYTHON ?? "python";

  const run = spawnSync(
    python,
    [script, "--provider", provider, "--amount", String(amountUsdc)],
    { encoding: "utf8" }
  );

  if (run.error) throw run.error;
  if (run.status !== 0) {
    throw new Error(
      `decide.py exited ${run.status}: ${run.stderr?.trim() || "(no stderr)"}`
    );
  }
  return JSON.parse(run.stdout) as Decision;
}

function reportDecision(d: Decision, provider: string): void {
  const rec = d.provider_record;
  console.log("");
  console.log("  ".padEnd(2) + "-".repeat(66));
  log.info(`provider ${shortAddr(provider)}`);
  log.info(
    `  public record        : ${rec.completed}/${rec.funded} completed` +
      (rec.funded === 0 ? "  (no history - unknown provider)" : "")
  );
  log.info(`  chain evidence only  : ${d.chain_only.p.toFixed(4)}  ${d.chain_only.decision}`);
  log.info(`  with memory          : ${d.with_memory.p.toFixed(4)}  ${d.with_memory.decision}`);
  for (const r of d.with_memory.rules_applied) {
    log.info(`      ${r.rule_id}  ${r.why}  delta ${r.delta_logit.toFixed(2)} (w=${r.weight})`);
  }
  log.info(
    `  memory changed it    : ${d.memory_changed_decision ? "YES - decision" : d.memory_changed_probability ? "probability only" : "no"}`
  );
  if (d.store) log.info(`  recalled from        : ${d.store}`);
  console.log("  ".padEnd(2) + "-".repeat(66));
  console.log("");
}

async function main(): Promise<void> {
  const provider = (
    process.env.PROVIDER_ADDRESS ?? DEFAULT_PROVIDER
  ).toLowerCase();
  const amountUsdc = Number(
    process.env.AMOUNT_USDC ?? String(DEFAULT_AMOUNT_USDC)
  );

  // 1. Decide, before anything touches the chain or costs anything.
  const decision = askPriors(provider, amountUsdc);
  reportDecision(decision, provider);

  if (!decision.hire) {
    log.info("decision is DO NOT HIRE - no job created, nothing spent.");
    return;
  }

  // 2. Only now connect a wallet.
  const priors = await AcpAgent.create({
    evmProvider: await PrivyAlchemyEvmProviderAdapter.create({
      walletAddress: requireEnv("PRIORS_WALLET_ADDRESS") as `0x${string}`,
      walletId: requireEnv("PRIORS_WALLET_ID"),
      signerPrivateKey: requireEnv("PRIORS_SIGNER_PRIVATE_KEY"),
      chains: [base],
    }),
  });

  const me = (await priors.getAddress()).toLowerCase();
  log.info(`buyer address: ${me}`);

  let settled = false;
  const finish = (why: string) => {
    if (settled) return;
    settled = true;
    log.info(why);
    void priors.stop();
  };

  priors.on("entry", async (session: JobSession, entry: JobRoomEntry) => {
    if (entry.kind === "message" && entry.from.toLowerCase() !== me) {
      log.job(session.jobId, `${shortAddr(entry.from)}: ${entry.content}`);
    }
    if (entry.kind !== "system") return;

    switch (entry.event.type) {
      case "budget.set": {
        const proposed = entry.event.amount;
        log.job(session.jobId, `provider proposed ${proposed} USDC`);
        // The budget is the provider's to set, but the spend is Priors'. A
        // provider that quotes above what was decided on is not the job that
        // was approved, so this refuses rather than silently overpaying.
        if (proposed > amountUsdc) {
          await session.sendMessage(
            `Proposed ${proposed} USDC exceeds the ${amountUsdc} USDC this job was approved for.`
          );
          await session.reject("budget above approved amount");
          finish("rejected: budget above what was approved");
          return;
        }
        try {
          await session.fetchJob();
          await session.fund();
          log.job(session.jobId, `funded ${proposed} USDC`);
        } catch (err) {
          log.error(`funding failed on job ${session.jobId}`, err);
          finish("stopping: funding failed");
        }
        break;
      }

      case "job.completed": {
        log.job(session.jobId, "COMPLETED");
        finish("done - the outcome is now on-chain for the resolver to record");
        break;
      }

      case "job.rejected": {
        log.job(session.jobId, `REJECTED: ${JSON.stringify(entry.event)}`);
        finish("done - job rejected");
        break;
      }

      case "job.expired": {
        log.job(session.jobId, "EXPIRED - escrow refunds to the buyer");
        finish("done - job expired");
        break;
      }

      default:
        log.job(session.jobId, entry.event.type);
    }
  });

  // 3. Start the agent. Nothing is dispatched before this: `start()` is what
  // hydrates sessions for jobs already in flight and begins delivering entry
  // events. Creating a job without it produces an on-chain job that this
  // process then ignores.
  await priors.start();
  log.info("listening");

  const shutdown = async (signal: NodeJS.Signals) => {
    log.info(`received ${signal}, shutting down`);
    await priors.stop();
    process.exit(0);
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);

  // 4. Resume before creating. After `start()` the SDK has rebuilt a session
  // for every active job this wallet is on, so a restart is already funding
  // whatever it left mid-flight. Creating another job here would quietly pile
  // a second one on top - which is exactly what happened on the first run of
  // this script, before this guard existed.
  const inFlight = priors.sessions.filter(
    (s) =>
      s.chainId === CHAIN_ID &&
      s.roles.includes("client") &&
      !["completed", "rejected", "expired"].includes(s.status)
  );
  if (inFlight.length > 0) {
    log.info(`resuming ${inFlight.length} in-flight job(s), creating none:`);
    for (const s of inFlight) {
      log.info(`  - job ${s.jobId}  status=${s.status}`);
    }
    return;
  }

  // 5. Create the job. `evaluatorAddress` omitted on purpose - see the header.
  try {
    const jobId = await priors.createJobByOfferingName(
      CHAIN_ID,
      OFFERING_NAME,
      provider,
      { text: SAMPLE_TEXT }
    );
    log.job(String(jobId), `created on Base, ${amountUsdc} USDC approved`);
  } catch (err) {
    log.error("createJobByOfferingName failed", err);
    await priors.stop();
    process.exitCode = 1;
  }
}

main().catch((err) => {
  log.error("fatal", err);
  process.exitCode = 1;
});
