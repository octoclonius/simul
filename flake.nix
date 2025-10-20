{
  inputs = {
    nix-develop-gha = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:nicknovitski/nix-develop";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = {
    self,
    nix-develop-gha,
    nixpkgs,
    ...
  }:
  let
    pkgs = nixpkgs.legacyPackages.aarch64-linux;
  in
  {
    packages.aarch64-linux.nix-develop-gha = nix-develop-gha.packages.aarch64-linux.default;
    devShells.aarch64-linux.default = pkgs.mkShell {
      packages = with pkgs.python312Packages; [
        pettingzoo
      ];
    };
  };
}
