"""Check built distributions without importing either numerical package.

Run after ``uv build --all-packages --out-dir dist``. This checks the actual
archives, including ownership and source bytes, rather than just build settings.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def single(paths: list[Path], description: str) -> Path:
    require(len(paths) == 1, f"Expected exactly one {description}; found {paths}")
    return paths[0]


def check_metadata(metadata: bytes, project: dict) -> None:
    fields = BytesParser().parsebytes(metadata)
    require(fields["Name"] == project["name"], "Distribution name differs from project")
    require(fields["Version"] == project["version"], "Distribution version differs from project")
    require(
        SpecifierSet(fields["Requires-Python"]) == SpecifierSet(project["requires-python"]),
        "Python requirement differs",
    )
    require(fields["License"] == "Apache-2.0", "Missing Apache-2.0 license declaration")
    requirements = [Requirement(value) for value in fields.get_all("Requires-Dist", [])]
    for extra in ("", *project.get("optional-dependencies", {})):
        active = [
            req
            for req in requirements
            if req.marker is None or req.marker.evaluate({"extra": extra})
        ]
        expected = [
            Requirement(value)
            for value in (
                project["dependencies"]
                + project.get("optional-dependencies", {}).get(extra, [])
            )
        ]

        def signature(req: Requirement) -> tuple:
            return canonicalize_name(req.name), str(req.specifier), tuple(sorted(req.extras)), req.url

        require(
            sorted(map(signature, active)) == sorted(map(signature, expected)),
            f"Unexpected dependencies for extra {extra!r}",
        )


def read_wheel(path: Path, project: dict, license_bytes: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), f"Duplicate archive members in {path.name}")
        require(
            all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                for name in names),
            f"Unsafe member path in {path.name}",
        )
        files = {name: archive.read(name) for name in names if not name.endswith("/")}
    metadata_names = [name for name in files if name.endswith(".dist-info/METADATA")]
    require(len(metadata_names) == 1, f"Expected one wheel metadata record in {path.name}")
    check_metadata(files[metadata_names[0]], project)
    licenses = [value for name, value in files.items() if name.endswith("/LICENSE")]
    require(licenses == [license_bytes], f"License missing or changed in {path.name}")
    return files


def check_physics_compatibility(project: dict) -> None:
    python = SpecifierSet(project["requires-python"])
    require(all(version in python for version in ("3.11", "3.12")), "Python support narrowed")
    require(all(version not in python for version in ("3.10", "3.13")), "Python support widened")
    requirements = {req.name: req for value in project["dependencies"] for req in [Requirement(value)]}
    for numpy, scipy in (("2.1.3", "1.14.1"), ("2.4.6", "1.17.1")):
        require(numpy in requirements["numpy"].specifier, f"NumPy {numpy} cannot resolve")
        require(scipy in requirements["scipy"].specifier, f"SciPy {scipy} cannot resolve")


def check_artifacts(repository: Path, directory: Path) -> dict:
    physics = repository / "packages/noether_physics"
    member = tomllib.loads((physics / "pyproject.toml").read_text())["project"]
    root = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
    license_bytes = (repository / "LICENSE").read_bytes()
    require((physics / "LICENSE").read_bytes() == license_bytes, "Package license differs")
    check_physics_compatibility(member)

    physics_wheel = single(sorted(directory.glob("noether_physics-*.whl")), "physics wheel")
    audit_wheel = single(sorted(directory.glob("noether_link_audit-*.whl")), "audit wheel")
    physics_files = read_wheel(physics_wheel, member, license_bytes)
    audit_files = read_wheel(audit_wheel, root, license_bytes)
    source = physics / "src"
    expected = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in sorted((source / "noether_physics").rglob("*.py"))
    }
    actual = {name: value for name, value in physics_files.items() if name.startswith("noether_physics/")}
    require(actual == expected, "Physics wheel source differs from the repository")
    require(
        not any("noether_physics" in PurePosixPath(name).parts for name in audit_files),
        "Link Audit wheel also owns noether_physics; uninstalling either could break the other",
    )
    require(
        not any("noether_link_audit" in PurePosixPath(name).parts for name in physics_files),
        "Physics wheel contains the Link Audit package",
    )
    require(
        "noether_link_audit/__init__.py" in audit_files,
        "Link Audit wheel is missing its import package",
    )

    sdist = single(sorted(directory.glob("noether_physics-*.tar.gz")), "physics source archive")
    expected_sdist = {
        # Hatchling 1.27 force-includes the containing repository's ignore file.
        # Require those exact public bytes, not an arbitrary extra archive member.
        ".gitignore": (repository / ".gitignore").read_bytes(),
        "pyproject.toml": (physics / "pyproject.toml").read_bytes(),
        "README.md": (physics / "README.md").read_bytes(),
        "LICENSE": license_bytes,
        **{f"src/{name}": value for name, value in expected.items()},
    }
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        require(len(names) == len(set(names)), "Duplicate source archive members")
        prefixes = {PurePosixPath(name).parts[0] for name in names}
        require(len(prefixes) == 1, "Source archive has more than one root")
        observed = {}
        for item in members:
            name = PurePosixPath(item.name)
            require(not name.is_absolute() and ".." not in name.parts, "Unsafe source member")
            require(item.isdir() or item.isfile(), "Source archive contains a link or special file")
            if item.isfile():
                stream = archive.extractfile(item)
                require(stream is not None, f"Cannot read source member {name}")
                with stream:
                    observed[PurePosixPath(*name.parts[1:]).as_posix()] = stream.read()
    require(observed.keys() == expected_sdist.keys() | {"PKG-INFO"}, "Unexpected source archive inventory")
    require(all(observed[name] == value for name, value in expected_sdist.items()), "Source archive differs")
    check_metadata(observed["PKG-INFO"], member)
    return {
        "status": "passed",
        "physics_wheel": physics_wheel.name,
        "audit_wheel": audit_wheel.name,
        "physics_sdist": sdist.name,
        "physics_source_files": sorted(expected),
        "physics_namespace_owner": "noether-physics",
        "dependency_metadata_admits": [
            {"numpy": "2.1.3", "scipy": "1.14.1"},
            {"numpy": "2.4.6", "scipy": "1.17.1"},
        ],
        "scope": "Distribution contents and dependency metadata; numerical tests run separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing freshly built distributions")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    print(json.dumps(check_artifacts(repository, args.directory), indent=2))


if __name__ == "__main__":
    main()
