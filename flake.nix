{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = {
    self,
    nixpkgs,
  }:
  let
    pkgs = nixpkgs.legacyPackages.aarch64-linux;
  in
  {
    devShells.aarch64-linux.default = pkgs.mkShell {
      packages = with pkgs.python312Packages; [
        pettingzoo
      ];
    };
  };
}
