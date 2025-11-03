// main.cpp — C++ driver that embeds Python and calls quantum_lab.py

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>   // _putenv_s

namespace py = pybind11;

int main() {
    try {
        // ---- YOUR PATHS (from your python -c output) ----
        const char* PY_BASE   = R"(C:\Python313)";  // base Python install (has Lib\encodings and DLLs)
        const char* VENV_SITE = R"(C:\Users\robyn\.vscode\Quantum\.venv\Lib\site-packages)";
        const char* PROJ_DIR  = R"(C:\Users\robyn\.vscode\Quantum)"; // contains quantum_lab.py

        // Environment for the embedded interpreter
        _putenv_s("PYTHONHOME", PY_BASE);       // stdlib comes from base Python
        _putenv_s("PYTHONPATH", VENV_SITE);     // third-party packages from venv
        _putenv_s("PYTHONUTF8", "1");
        _putenv_s("MPLBACKEND", "Agg");

        // Make sure Python DLLs are discoverable
        std::string path = std::string(PY_BASE) + ";" + std::string(PY_BASE) + R"(\DLLs)";
        if (const char* old = std::getenv("PATH")) path += std::string(";") + old;
        _putenv_s("PATH", path.c_str());

        // Start Python
        py::scoped_interpreter guard{};

        // Build strings *before* passing to pybind11 (avoid temporaries)
        std::string projDir   = PROJ_DIR;
        std::string sitePkgs  = VENV_SITE;
        std::string stdlibDir = std::string(PY_BASE) + R"(\Lib)";

        auto sys = py::module::import("sys");
        sys.attr("path").attr("append")(projDir);
        sys.attr("path").attr("append")(sitePkgs);
        sys.attr("path").attr("append")(stdlibDir);

        // Optional preflight: show whether token is set
        std::cout << "IBM_QUANTUM_TOKEN set? " 
                  << (std::getenv("IBM_QUANTUM_TOKEN") ? "yes" : "no") << "\n";

        auto m = py::module::import("quantum_lab");

        // ===== Example 1: local timing
        std::cout << "Running py_measure_runtime_vs_size(2, 5, 512) ...\n";
        py::tuple tup = m.attr("py_measure_runtime_vs_size")(2, 5, 512);
        auto ns     = tup[0].cast<std::vector<int>>();
        auto sqrtNs = tup[1].cast<std::vector<double>>();
        auto iters  = tup[2].cast<std::vector<int>>();
        auto secs   = tup[3].cast<std::vector<double>>();
        std::cout << "n  sqrtN  iters  secs\n";
        for (size_t i = 0; i < ns.size(); ++i)
            std::cout << ns[i] << "  " << sqrtNs[i] << "  " << iters[i] << "  " << secs[i] << "\n";

        // ===== Example 2: IBM echo (print full traceback on failure)
        std::cout << "\nRunning py_run_echo(depth=8, shots=1024, n_qubits=5, basis='x', seed=42) ...\n";
        try {
            py::dict echo = m.attr("py_run_echo")(8, 1024, 5, "x", 42);
            std::cout << "Echo: " << py::str(echo).cast<std::string>() << "\n";
        } catch (py::error_already_set& e) {                 // NOTE: non-const
            std::cerr << "Python error in py_run_echo:\n" << e.what() << "\n";
            e.restore();                                     // re-raise in Python
            PyErr_Print();                                   // full traceback to stderr
            return 1;
        }
    }
    catch (py::error_already_set& e) {                       // NOTE: non-const
        std::cerr << "Python error:\n" << e.what() << "\n";
        e.restore();
        PyErr_Print();
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "C++ error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
