{pkgs}: {
  deps = [
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
