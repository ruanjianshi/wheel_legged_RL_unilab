#ifndef POLICY_CONFIG_H
#define POLICY_CONFIG_H

#include <yaml-cpp/yaml.h>

namespace hightorque
{
    namespace config
    {
        class PolicyConfig
        {
            public:
                class Policy
                {
                    public:
                        std::string name;
                };
                class Motor
                {
                    public:
                        std::string name;
                        int dofs; // 机器人自由度

                };
        };
    }
}
#endif
