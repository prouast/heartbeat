//
//  Heartbeat.hpp
//  Heartbeat
//
//  Created by Philipp Rouast on 4/06/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#ifndef Heartbeat_hpp
#define Heartbeat_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include "RPPG.hpp"

using namespace std;

class Heartbeat {
    
public:
    
    Heartbeat(int argc_, char * argv_[], bool switches_on_ = false);
    ~Heartbeat(){}
    
    string get_arg(int i);
    string get_arg(string s);
    
private:
    
    int argc;
    vector<string> argv;
    
    bool switches_on;
    map<string, string> switch_map;
    
};

#endif /* Heartbeat_hpp */
