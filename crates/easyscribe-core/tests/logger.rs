//! Startup installs the file logger before the payload is prepared and then
//! again from `app::startup`; the second call must not create another log.

#[test]
fn setup_is_idempotent_and_keeps_the_first_log_file() {
    let dir = tempfile::tempdir().unwrap();
    easyscribe_core::logger::setup(dir.path()).unwrap();
    easyscribe_core::logger::setup(dir.path()).unwrap();
    log::info!("logger idempotence check");
    log::logger().flush();

    let logs: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("log"))
        .collect();
    assert_eq!(logs.len(), 1, "a second setup must not rotate or add a log");

    let text = std::fs::read_to_string(&logs[0]).unwrap();
    assert!(text.contains("logger idempotence check"));
}
