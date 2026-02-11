# Install PyMC


## Virtual Env
 Unlike the docs I prefer, Pyhthon -m venv Virtual Environment which works well in most environments including MAC, Unix, Windows, Docker (linux). The following has been successful so far.

Create the venv, for example. In your project directory.

```sh
> python -m venv myvenv
> # MAC or Linux
> source myenv/bin/activate
> # PC Windoes
> source myenv/Scripts/activate
(myenv) > pip install pymc
```

# Issue NUTS C-Compiler Error

```text
CompileError: Compilation failed (return status=1) 64/tmprod8vllb/mod.cpp:7:10: fatal error: 'vector' file not found
```
On the Mac, the first run failed. Based on some online searches, I uninstalled XCode build tools and re-installed. After the installation above (venv and pip install), I uninstalled Xcode build tools and PyMC NUTS worked just fine.

```sh
> sudo rm -rf /Library/Developer/CommandLineTools # uninstall build tools
> xcode-select --install # install build tools
```

After this PyMC began working as expected
