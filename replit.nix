{pkgs}: {
  deps = [
    pkgs.wget
    pkgs.libGLU
    pkgs.libGL
    pkgs.postgresql
    pkgs.openssl
  ];
}
