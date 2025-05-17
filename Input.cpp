#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm> // for std::remove_if
#include "Input.h"

using namespace std;

// Helper function to extract the first double from a string
double extractDouble(const std::string& text) {
    static const string digits = "0123456789.+-Ee";
    size_t pos = text.find_first_of(digits);
    if (pos != string::npos) {
        stringstream ss(text.substr(pos));
        double val = 0.0;
        ss >> val;
        return val;
    }
    return 0.0;
}

void input(const std::string& in_path) {

    const string filename_input = in_path + "BTE.in";
    ifstream input_stream(filename_input);

    if (!input_stream.is_open()) {
        cerr << "Error: Unable to open the input file \"" << filename_input << "\"." << endl;
        return;
    }

    string line;
    while (getline(input_stream, line)) {
        // Remove whitespace from line start and end
        line.erase(line.begin(), find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !isspace(ch);
        }));
        line.erase(find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !isspace(ch);
        }).base(), line.end());

        if (line.empty() || line[0] == '#') continue; // skip blank lines or comments

        auto parse_param = [&](const string& param, auto& variable, bool is_int = false) {
            size_t pos = line.find(param);
            if (pos != string::npos) {
                size_t eq_pos = line.find('=', pos);
                if (eq_pos != string::npos) {
                    string val_str = line.substr(eq_pos + 1);
                    double val = extractDouble(val_str);
                    if (is_int)
                        variable = static_cast<int>(val);
                    else
                        variable = val;
                    return true;
                }
            }
            return false;
        };

        bool read_any = false;
        read_any |= parse_param("dt", dt);
        read_any |= parse_param("Natom", Natom, true);
        read_any |= parse_param("Nstep", Nstep, true);
        read_any |= parse_param("T_target", T_target);
        read_any |= parse_param("d_T", d_T);
        read_any |= parse_param("Nz", Nz, true);
        read_any |= parse_param("Lx", Lx);
        read_any |= parse_param("Ly", Ly);
        read_any |= parse_param("Lz", Lz);
        read_any |= parse_param("V_FC", V_FC);
        read_any |= parse_param("Nout", Nout);
        read_any |= parse_param("Navg", Navg);
        read_any |= parse_param("Nprint", Nprint);
        read_any |= parse_param("W", W);
        read_any |= parse_param("Tstart", Tstart, true);
        read_any |= parse_param("Tend", Tend, true);
        read_any |= parse_param("Tincrement", Tincrement, true);

        // Optional: if (!read_any) cout << "Warning: unrecognized line: " << line << endl;
    }

    input_stream.close();
}
