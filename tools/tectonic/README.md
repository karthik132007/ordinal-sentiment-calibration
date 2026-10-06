# Portable LaTeX compiler

The study used Tectonic 0.17.0. The platform-specific executable is excluded from Git; the compiled paper and source files are included.

For Linux x86_64, from the repository root:

```bash
mkdir -p tools/tectonic tmp
curl -fL 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz' \
  -o tmp/tectonic.tar.gz
tar -xzf tmp/tectonic.tar.gz -C tools/tectonic
tools/tectonic/tectonic --version
```

For other operating systems, download the matching executable from the [official release](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0) and place it at `tools/tectonic/tectonic`. Initial compilation may download LaTeX resources. See the main README for the build and reproduction commands.
