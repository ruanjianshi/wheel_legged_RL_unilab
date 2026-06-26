#ifndef MIXLEVEL_CONTROLLER_H_
#define MIXLEVEL_CONTROLLER_H_

#include "highlevel_controller.h"
#include "lowlevel_controller.h"

namespace hightorque
{
    /**
     * @brief 混合控制节点：组合高低级控制器，统一对外接口
     */
    class MixlevelControllerNode : public LevelControllerNode
    {
        public:
            /** ======================== 构造与析构 ======================== */

            /**
             * @brief 构造函数：注入高低级控制节点
             * @param low  低级控制节点共享指针
             * @param high 高级控制节点共享指针
             */
            MixlevelControllerNode(std::shared_ptr<LowlevelControllerNode> low,
                                   std::shared_ptr<HighlevelControllerNode> high);

            /**
             * @brief 析构函数：清理资源
             */
            ~MixlevelControllerNode();

            /** ======================== 覆写接口 ======================== */

            /**
             * @brief 初始化所有子节点（覆盖 LevelControllerNode）
             * @return true 表示初始化成功
             */
            bool init() override;

            /**
             * @brief 启动所有子节点（覆盖 LevelControllerNode）
             * @return true 表示启动成功
             */
            bool start() override;

            /**
             * @brief 检查是否请求退出（覆盖 LevelControllerNode）
             * @note 低级或高级控制器任一请求退出即返回 true
             * @return true 表示应退出主循环
             */
            bool get_is_shutdowm() override
            {
                if (low_->get_is_shutdowm() || high_->get_is_shutdowm())
                {
                    ROS_INFO("low: %d high: %d", low_->get_is_shutdowm(), high_->get_is_shutdowm());
                    return true;
                }
                return false;
            }

        private:
            std::shared_ptr<LowlevelControllerNode> low_;   // 低级控制节点
            std::shared_ptr<HighlevelControllerNode> high_; // 高级控制节点
    };
};
#endif

