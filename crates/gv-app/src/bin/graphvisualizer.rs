fn main() -> anyhow::Result<()> {
    env_logger::init();
    gv_app::run(gv_app::Cli::parse_from_env()?)
}
