//! Windows executable resources: the EasyScribe icon, version information and
//! a per-monitor DPI-aware manifest. The app ships as one file, so the icon
//! must live in the binary rather than beside it.

fn main() {
    println!("cargo:rerun-if-changed=../../assets/EasyScribe.ico");
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../../assets/EasyScribe.ico")
            .set("ProductName", "EasyScribe")
            .set("FileDescription", "EasyScribe")
            .set("CompanyName", "EasyScribe")
            .set("LegalCopyright", "MIT licensed")
            .set("FileVersion", "3.2.3.0")
            .set("ProductVersion", "3.2.3")
            .set_manifest(
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="1.0.0.0" processorArchitecture="*" name="EasyScribe" type="win32"/>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security><requestedPrivileges>
      <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
    </requestedPrivileges></security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
    </windowsSettings>
  </application>
</assembly>"#,
            );
        if let Err(error) = res.compile() {
            panic!("could not compile Windows resources: {error}");
        }
    }
}
