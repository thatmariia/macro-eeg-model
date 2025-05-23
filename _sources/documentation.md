# Documentation

**That's where you are now 😊**

The documentation is built with Sphinx. 
It provides the API reference and the user guide.

## Open documentation

To open the documentation, use this command:

* On macOS:
```sh
open docs/build/html/index.html
```

* On Linux:
```sh
firefox docs/build/html/index.html
```
(replace `firefox` with `google-chrome` or another browser if necessary).
If running on Ubuntu, you may simply run 
```sh
xdg-open docs/build/html/index.html
```

* On Windows:
```sh
start docs/build/html/index.html
```

## Regenerate documentation

To regenerate the documentation, use this command:
```sh
make -C docs html
```