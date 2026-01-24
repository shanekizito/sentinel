fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    tonic_build::configure()
        .build_server(false)
        .compile(
            &["../sentinel-orchestrator/proto/sentinel.proto"],
            &["../sentinel-orchestrator/proto"],
        )?;
    Ok(())
}
