//! Command-line overrides.
//!
//! The original could only be reconfigured by editing `settings.json` and
//! relaunching, which made benchmarking across the seven datasets tedious.
//! Anything set here wins over the settings file.
//!
//! Hand-rolled rather than `clap`: five options, `--key value` and `--key=value`,
//! not worth a proc-macro dependency in the one crate that already pulls in
//! wgpu, winit and egui.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use gv_config::AppConfig;
use gv_gui::LayoutChoice;

pub const USAGE: &str = "\
graphvisualizer — force-directed graph layout on the GPU

USAGE:
    graphvisualizer [OPTIONS] [EDGE_LIST]

OPTIONS:
    -s, --settings <PATH>   Settings file (default: settings.json)
    -e, --edge-input <PATH> Edge list to load, overriding `edgeInput`
    -n, --nodes-input <PATH> Knowledge-graph node sidecar (nodes.tsv): colour by
                            kind and enable search, overriding `nodesInput`
    -l, --layout <NAME>     gpu | gpu-barnes-hut | cpu | barnes-hut | random
        --seed <N>          RNG seed for the initial scatter (default: 0)
        --headless <STEPS>  Run without a window for STEPS steps, then report
    -h, --help              Print this message
";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cli {
    /// Settings file to read. Defaults to `settings.json`; a missing file is
    /// not an error, the defaults are used.
    pub settings: Option<PathBuf>,
    /// Edge list to load, overriding `edgeInput`.
    pub edge_input: Option<PathBuf>,
    /// Knowledge-graph node sidecar (`nodes.tsv`), enabling colour-by-kind and
    /// search. Overrides `nodesInput`.
    pub nodes_input: Option<PathBuf>,
    pub layout: Option<LayoutChoice>,
    /// RNG seed for the initial scatter, so runs are reproducible.
    pub seed: Option<u64>,
    /// Run without a window for `n` steps, then report. Used to compare the
    /// CPU and GPU paths and to benchmark.
    pub headless_steps: Option<u32>,
    /// Set by `--help`; the caller prints [`USAGE`] and exits successfully.
    pub help: bool,
}

impl Cli {
    pub fn parse_from_env() -> Result<Self> {
        Self::parse(std::env::args().skip(1))
    }

    /// Parses already-split arguments, program name excluded.
    pub fn parse(args: impl IntoIterator<Item = impl Into<String>>) -> Result<Self> {
        let mut cli = Self::default();
        let mut args = args.into_iter().map(Into::into).peekable();

        while let Some(arg) = args.next() {
            // `--key=value` and `--key value` are both accepted; splitting here
            // means each option below only has to deal with the latter.
            let (flag, inline) = match arg.split_once('=') {
                Some((flag, value)) if flag.starts_with('-') => {
                    (flag.to_owned(), Some(value.to_owned()))
                }
                _ => (arg.clone(), None),
            };

            let mut value = |flag: &str| -> Result<String> {
                inline
                    .clone()
                    .or_else(|| args.next())
                    .with_context(|| format!("{flag} expects a value"))
            };

            match flag.as_str() {
                "-h" | "--help" => cli.help = true,
                "-s" | "--settings" => cli.settings = Some(value(&flag)?.into()),
                "-e" | "--edge-input" => cli.edge_input = Some(value(&flag)?.into()),
                "-n" | "--nodes-input" => cli.nodes_input = Some(value(&flag)?.into()),
                "-l" | "--layout" => {
                    let raw = value(&flag)?;
                    cli.layout = Some(LayoutChoice::from_str(&raw).map_err(anyhow::Error::msg)?);
                }
                "--seed" => {
                    let raw = value(&flag)?;
                    cli.seed = Some(raw.parse().with_context(|| format!("--seed {raw:?}"))?);
                }
                "--headless" => {
                    let raw = value(&flag)?;
                    cli.headless_steps =
                        Some(raw.parse().with_context(|| format!("--headless {raw:?}"))?);
                }
                other if other.starts_with('-') => bail!("unknown option {other:?}\n\n{USAGE}"),
                positional => {
                    if cli.edge_input.is_some() {
                        bail!("unexpected second edge list {positional:?}");
                    }
                    cli.edge_input = Some(positional.into());
                }
            }
        }

        Ok(cli)
    }

    /// Applies the overrides that shadow settings-file keys.
    pub fn apply_to(&self, config: &mut AppConfig) {
        if let Some(edge_input) = &self.edge_input {
            config.edge_input = edge_input.clone();
        }
        if let Some(nodes_input) = &self.nodes_input {
            config.nodes_input = Some(nodes_input.clone());
        }
    }

    pub fn is_headless(&self) -> bool {
        self.headless_steps.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::parse(args.iter().copied()).expect("should parse")
    }

    #[test]
    fn no_arguments_overrides_nothing() {
        assert_eq!(parse(&[]), Cli::default());
    }

    #[test]
    fn usage_lists_every_layout_the_picker_offers() {
        // This drifted the moment a variant was added: the help text kept
        // advertising the old four while `--layout gpu-barnes-hut` already
        // worked. The listing is the only place a user finds the name.
        for choice in gv_gui::LayoutChoice::ALL {
            assert!(
                USAGE.contains(choice.slug()),
                "{:?} ({}) is missing from USAGE",
                choice,
                choice.slug()
            );
        }
    }

    #[test]
    fn accepts_space_separated_values() {
        let cli = parse(&["--layout", "cpu", "--seed", "42", "--headless", "500"]);
        assert_eq!(cli.layout, Some(LayoutChoice::FrCpu));
        assert_eq!(cli.seed, Some(42));
        assert_eq!(cli.headless_steps, Some(500));
    }

    #[test]
    fn accepts_equals_separated_values() {
        let cli = parse(&["--layout=barnes-hut", "--seed=7"]);
        assert_eq!(cli.layout, Some(LayoutChoice::FrBarnesHut));
        assert_eq!(cli.seed, Some(7));
    }

    #[test]
    fn accepts_short_flags() {
        let cli = parse(&["-l", "gpu", "-e", "array.edges", "-s", "custom.json"]);
        assert_eq!(cli.layout, Some(LayoutChoice::FrGpu));
        assert_eq!(cli.edge_input, Some(PathBuf::from("array.edges")));
        assert_eq!(cli.settings, Some(PathBuf::from("custom.json")));
    }

    #[test]
    fn a_bare_positional_is_the_edge_list() {
        let cli = parse(&["data/facebook_combined.txt"]);
        assert_eq!(cli.edge_input, Some(PathBuf::from("data/facebook_combined.txt")));
    }

    /// A path containing `=` must not be mistaken for an inline value; only
    /// arguments that start with `-` are split.
    #[test]
    fn a_positional_containing_equals_is_not_split() {
        let cli = parse(&["data/graph=v2.edges"]);
        assert_eq!(cli.edge_input, Some(PathBuf::from("data/graph=v2.edges")));
    }

    #[test]
    fn edge_input_flag_and_positional_conflict() {
        let error = Cli::parse(["-e", "a.edges", "b.edges"]).unwrap_err();
        assert!(error.to_string().contains("second edge list"), "{error}");
    }

    #[test]
    fn rejects_unknown_options() {
        let error = Cli::parse(["--nodes=5"]).unwrap_err();
        assert!(error.to_string().contains("unknown option"), "{error}");
    }

    #[test]
    fn rejects_an_unparsable_layout_name() {
        let error = Cli::parse(["--layout", "kd-tree"]).unwrap_err();
        assert!(error.to_string().contains("unknown layout"), "{error}");
    }

    #[test]
    fn rejects_a_non_numeric_seed() {
        let error = Cli::parse(["--seed", "abc"]).unwrap_err();
        assert!(error.to_string().contains("--seed"), "{error}");
    }

    #[test]
    fn reports_a_missing_trailing_value() {
        let error = Cli::parse(["--seed"]).unwrap_err();
        assert!(error.to_string().contains("expects a value"), "{error}");
    }

    #[test]
    fn help_is_recognised_and_does_not_error() {
        assert!(parse(&["--help"]).help);
        assert!(parse(&["-h"]).help);
    }

    #[test]
    fn edge_input_overrides_the_settings_file() {
        let mut config = AppConfig::default();
        parse(&["-e", "other.edges"]).apply_to(&mut config);
        assert_eq!(config.edge_input, PathBuf::from("other.edges"));
    }

    #[test]
    fn absent_edge_input_leaves_the_settings_value_alone() {
        let mut config = AppConfig::default();
        let original = config.edge_input.clone();
        parse(&["--seed", "1"]).apply_to(&mut config);
        assert_eq!(config.edge_input, original);
    }

    #[test]
    fn headless_is_detected_only_when_steps_are_given() {
        assert!(parse(&["--headless", "10"]).is_headless());
        assert!(!parse(&[]).is_headless());
    }

    /// Every option in USAGE must actually parse — the two drift otherwise.
    #[test]
    fn every_documented_long_option_is_accepted() {
        for line in USAGE.lines() {
            for word in line.split_whitespace() {
                let Some(flag) = word.strip_suffix(',').or(Some(word)) else { continue };
                if !flag.starts_with("--") {
                    continue;
                }
                let error = Cli::parse([flag]).unwrap_err_or_ok();
                assert!(
                    !error.contains("unknown option"),
                    "{flag} appears in USAGE but is not parsed"
                );
            }
        }
    }

    /// Helper: the error text, or an empty string when parsing succeeded.
    trait UnwrapErrOrOk {
        fn unwrap_err_or_ok(self) -> String;
    }

    impl UnwrapErrOrOk for Result<Cli> {
        fn unwrap_err_or_ok(self) -> String {
            self.err().map(|e| e.to_string()).unwrap_or_default()
        }
    }
}
