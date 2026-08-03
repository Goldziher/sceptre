//! sceptre command-line interface.
//!
//! A thin orchestration layer over the `sceptre` library: it parses arguments,
//! builds a configuration, and drives the pipeline. Diagnostics go to stderr via
//! tracing; structured data is written to stdout.

mod cli;
mod output;
mod overrides;
mod style;
mod timing;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = cli::Cli::parse();
    cli.init_tracing();
    cli.run()
}
