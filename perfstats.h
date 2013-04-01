/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef PERFSTATS_H
#define PERFSTATS_H

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <ctime>

struct PerfStats {
    enum Type { TIME, COUNT, PERCENTAGE };
    struct Stats {
        std::vector<double> data;
        Type type;
        double sum() const { return std::accumulate(data.begin(), data.end(), 0.0); }
        double average() const { return sum()/std::max(data.size(), size_t(1)); }
        double max() const { return *std::max_element(data.begin(), data.end()); }
        double min() const { return *std::min_element(data.begin(), data.end()); }
    };

    std::map<std::string, Stats> stats;
    double last;

    static double get_time() {
        return double(std::clock())/CLOCKS_PER_SEC;
    }

    void sample(const std::string& key, double t, Type type = COUNT) {
        Stats& s = stats[key];
        s.data.push_back(t);
        s.type = type;
    }
    double start(void){
        last = get_time();
        return last;
    }
    double sample(const std::string &key){
        const double now = get_time();
        sample(key, now - last, TIME);
        last = now;
        return now;
    }
    const Stats& get(const std::string& key) const { return stats.find(key)->second; }
    void reset(void) { stats.clear(); }
    void reset(const std::string & key);
    void print(std::ostream& out = std::cout) const;
};

inline void PerfStats::reset(const std::string & key){
    std::map<std::string,Stats>::iterator s = stats.find(key);
    if(s != stats.end())
        s->second.data.clear();
}

inline void PerfStats::print(std::ostream& out) const {
    for (std::map<std::string,Stats>::const_iterator it=stats.begin(); it!=stats.end(); it++){
        out << it->first << ":";
        out << std::string("\t\t\t").substr(0, 3 - ((it->first.size()+1) >> 3));
        switch(it->second.type){
        case TIME: {
            out << it->second.average()*1000.0 << "\t(max = " << it->second.max()*1000 << ")\n";
        } break;
        case COUNT: {
            out << it->second.average() << "\t(max = " << it->second.max() << " )\n";
        } break;
        case PERCENTAGE: {
            out << it->second.average()*100.0 << "%\t(max = " << it->second.max()*100 << " %)\n";
        } break;
        }
    }
}

extern PerfStats Stats;

#endif // PERFSTATS_H
