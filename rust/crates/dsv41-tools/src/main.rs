mod bench;
mod checks;
mod exact;
mod fake_server;
mod kv_shard;
mod prefix_fanout;
mod selftest;
mod spill_check;
mod validate;

use clap::{Parser, Subcommand};
use std::process;

#[derive(Parser)]
#[command(
    name = "dsv41-tools",
    version,
    about = "High-performance Rust benchmark, validation, and testing suite for DeepSeek-V4.1-Flash"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Streaming prefill/decode/multi-stream/needle benchmark & JSON diff (--compare)
    Bench(bench::BenchArgs),
    /// Bit-exactness harness (record & compare token hashes across engine images)
    Exact(exact::ExactArgs),
    /// End-to-end validation (text, thinking, tools, image, long prompt, concurrent==solo)
    Validate(validate::ValidateArgs),
    /// Fast correctness smoke test (health, stop, stream, tools, determinism, thinking)
    Smoke(validate::SmokeArgs),
    /// Needle-in-a-haystack check at ~195k+ tokens
    Needle(checks::NeedleArgs),
    /// Verify tool calls return typed JSON arrays
    Toolcheck(checks::ToolcheckArgs),
    /// Anthropic /v1/messages surface check (incl. streaming tool_use & chunked body)
    Anthropiccheck(checks::AnthropicArgs),
    /// End-of-turn check on French coding prompts + P(end of turn)
    EndOfTurn(checks::EndOfTurnArgs),
    /// Prompt-state reuse check under interleaved shared-system conversations
    PromptReuse(checks::PromptReuseArgs),
    /// Short request latency during a long cold prefill
    PrefillYield(checks::PrefillYieldArgs),
    /// Common-system-prompt fan-out duplicate prefill token check
    PrefixFanout(prefix_fanout::PrefixFanoutArgs),
    /// Broken-pair policy (exit 75) verification
    PairCheck(prefix_fanout::PairCheckArgs),
    /// Resume-from-spill benchmark (seed / resume / report)
    ResumeBench(prefix_fanout::ResumeBenchArgs),
    /// Spill tier end-to-end verification (record / run / drop / estimate)
    SpillCheck(spill_check::SpillCheckArgs),
    /// KV sharding end-to-end verification (record / run)
    KvShardCheck(kv_shard::KvShardArgs),
    /// High-throughput mock TensorFold server for offline testing
    FakeServer(fake_server::FakeServerArgs),
    /// Offline self-test of all validation tools against fake-server
    Selftest(selftest::SelftestArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::Bench(a) => bench::run(a).await,
        Cmd::Exact(a) => exact::run(a).await,
        Cmd::Validate(a) => validate::run_validate(a).await,
        Cmd::Smoke(a) => validate::run_smoke(a).await,
        Cmd::Needle(a) => checks::run_needle(a).await,
        Cmd::Toolcheck(a) => checks::run_toolcheck(a).await,
        Cmd::Anthropiccheck(a) => checks::run_anthropic(a).await,
        Cmd::EndOfTurn(a) => checks::run_end_of_turn(a).await,
        Cmd::PromptReuse(a) => checks::run_prompt_reuse(a).await,
        Cmd::PrefillYield(a) => checks::run_prefill_yield(a).await,
        Cmd::PrefixFanout(a) => prefix_fanout::run_prefix_fanout(a).await,
        Cmd::PairCheck(a) => prefix_fanout::run_pair_check(a).await,
        Cmd::ResumeBench(a) => prefix_fanout::run_resume_bench(a).await,
        Cmd::SpillCheck(a) => spill_check::run(a).await,
        Cmd::KvShardCheck(a) => kv_shard::run(a).await,
        Cmd::FakeServer(a) => fake_server::run(a).await,
        Cmd::Selftest(a) => selftest::run(a).await,
    };
    process::exit(code);
}
