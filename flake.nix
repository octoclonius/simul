{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }:
  let
    mkDevShell = pkgs:
      pkgs.mkShell {
        packages = with pkgs.python312Packages; [
          pettingzoo
          pylint
        ];
      };
    supportedSystems = [
      "aarch64-linux"
      "x86_64-linux"
    ];
  in
  {
    devShells = nixpkgs.lib.genAttrs supportedSystems (system: {
      default = mkDevShell nixpkgs.legacyPackages.${system};
    });
  };
}
