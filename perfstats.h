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
