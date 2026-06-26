#ifndef HT_FSM_H_
#define HT_FSM_H_

#include "sim2real/base_controller_interface.h"
#include "sim2real/default_controller_interface.h"
#include "sim2real/custom_controller_interface.h"
#include "sim2real/protection_controller_interface.h"
#include "sim2real/remote_controller_interface.h"
#include "sim2real/zerocalibration_controller_interface.h"
#include "sim2real/teach_controller_interface_n.h"
#include "sim2real/develop_controller_interface.h"
#include "livelybot_logger/logger_interface.h"

#include <boost/msm/front/state_machine_def.hpp>
#include <boost/msm/back/state_machine.hpp>
#include <boost/msm/front/functor_row.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/not.hpp>
namespace hightorque
{
    enum class FsmNodeType
    {
        INIT,                          // 0
        ERROR,                         // 1
        CANDIDATE_DEFAULT,             // 2
        CANDIDATE_CUSTOM,              // 3
        CANDIDATE_REMOTE,              // 4
        EXEC_DEFAULT,                  // 5
        EXEC_CUSTOM,                   // 6
        EXEC_REMOTE,                   // 7
        PROTECTION_SHUTDOWN,           // 8
        CANDIDATE_CALIBRATION,         // 9
        EXEC_CALIBRATING,              // 10
        EXEC_CALIBRATION_SUCCESSFULLY, // 11
        EXEC_CALIBRATION_FAILED,       // 12
        CANDIDATE_TEACHING,            // 13
        EXEC_TEACHING,                 // 14
        CANDIDATE_DEVELOP,             // 15
        EXEC_DEVELOP                   // 16
    };
    using namespace boost::msm::front;
    // 事件定义
    struct TickEvt
    {
    }; // 周期心跳事件
    struct LastOptionEvt
    {
    }; // 上一个选项
    struct NextOptionEvt
    {
    }; // 下一个选项
    struct ConfirmEvt
    {
    }; // 确认
    struct ProtectionShutdownEvt
    {
    }; // 保护关机
    struct CalibrationSuccessEvt
    {
    }; // 校0成功
    struct CalibrationFailedEvt
    {
    }; // 校0失败
    struct StartEvt
    {
    }; // 开始
    struct DefaultModeEvt
    {
    }; // 进入默认模式

    class MasterNode;

    // FSM 前端定义
    struct ModeFsmDef : state_machine_def<ModeFsmDef>
    {
        public:
            MasterNode* master;
            ModeFsmDef(MasterNode* m) : master(m) {}
            ModeFsmDef() : master(nullptr) {}

            static void print(std::string s)
            {
                static std::string status = "ExecDefault";
                if (s != status)
                {
                    ROS_INFO("from %s to %s", status.c_str(), s.c_str());
                }
                status = s;
            }

            FsmNodeType getState() const
            {
                return state;
            }

            // 状态声明
            // 定义状态机的状态
            struct Init : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::INIT;
                        print("Init");
                    }
            };
            struct CandidateDefault : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::CANDIDATE_DEFAULT;
                        print("CandidateDefault");
                    }
            };
            struct CandidateCustom : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::CANDIDATE_CUSTOM;
                        print("CandidateCustom");
                    }
            };
            struct CandidateRemote : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::CANDIDATE_REMOTE;
                        print("CandidateRemote");
                    }
            };
            struct CandidateCalibration : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::CANDIDATE_CALIBRATION;
                        print("CandidateCalibration");
                    }
            };
            struct CandidateTeaching : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::CANDIDATE_TEACHING;
                        print("CandidateTeaching");
                    }
            };
            struct CandidateDevelop : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::CANDIDATE_DEVELOP;
                        print("CandidateDevelop");
                    }
            };

            struct ExecDefault : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_DEFAULT;
                        print("ExecDefault");
                    }
            };
            struct ExecCustom : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_CUSTOM;
                        print("ExecCustom");
                    }
            };
            struct ExecRemote : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_REMOTE;
                        print("ExecRemote");
                    }
            };
            struct ExecCalibrating : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_CALIBRATING;
                        print("ExecCalibrating");
                    }
            };
            struct ExecCalibrationSuccess : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_CALIBRATION_SUCCESSFULLY;
                        print("ExecCalibrationSuccess");
                    }
            };
            struct ExecCalibrationFailed : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_CALIBRATION_FAILED;
                        print("ExecCalibrationFailed");
                    }
            };
            struct ExecTeaching : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_TEACHING;
                        print("ExecTeaching");
                    }
            };
            struct ExecDevelop : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::EXEC_DEVELOP;
                        print("ExecDevelop");
                    }
            };

            struct ProtectionShutdown : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::PROTECTION_SHUTDOWN;
                        print("ProtectionShutdown");
                    }
            };
            struct Error : state<>
            {
                    template <class Event, class FSM>
                    void on_entry(Event const&, FSM& fsm)
                    {
                        fsm.state = FsmNodeType::ERROR;
                        print("Error");
                    }
            };

            // 动作和 Guard
            struct ActSetMotorZero
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        for (auto motor : fsm.master->robotPtr_->Motors)
                        {
                            motor->fresh_cmd_int16(0, 0, 0,
                                                   0, 0, 0,
                                                   0, 0, 0);
                        }
                        fsm.master->robotPtr_->motor_send_2();
                    }
            };
            struct ActSendMotor
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        fsm.master->robotPtr_->motor_send_2();
                    }
            };
            struct ActRecordTime
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        fsm.master->stateEntryTime_ = ros::Time::now();
                    }
            };
            struct ActCheckCalibration
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        int status = fsm.master->ctrlInterfacePtr_->get_parameters();
                        if (status == 1)
                        {
                            fsm.process_event(CalibrationSuccessEvt());
                            fsm.master->stateEntryTime_ = ros::Time::now();
                        }
                        else if (status == 2)
                        {
                            fsm.process_event(CalibrationFailedEvt());
                            fsm.master->stateEntryTime_ = ros::Time::now();
                        }
                    }
            };

            struct ActResetDefaultCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        f.master->ctrlInterfacePtr_.reset(
                            new DefaultControllerInterface(f.master->robotPtr_));
                    }
            };
            struct ActResetCustomCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        f.master->ctrlInterfacePtr_.reset(
                            new CustomControllerInterface(f.master->robotPtr_));
                    }
            };
            struct ActResetRemoteCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        f.master->ctrlInterfacePtr_.reset(
                            new RemoteControllerInterface(f.master->robotPtr_));
                    }
            };
            struct ActResetZeroCalibCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        f.master->ctrlInterfacePtr_.reset(
                            new ZeroCalibrationControllerInterface(f.master->robotPtr_));
                    }
            };
            struct ActResetTeachCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        f.master->ctrlInterfacePtr_.reset(
                            new teach::TeachControllerInterface(f.master->robotPtr_));
                    }
            };
            struct ActResetDevelopCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        // 恢复DevelopControllerInterface
                        f.master->ctrlInterfacePtr_.reset(
                            new DevelopControllerInterface(f.master->robotPtr_));
                        // ROS_WARN("DevelopControllerInterface is disabled due to compilation issues");
                    }
            };
            struct ActResetProtectionCtrl
            {
                    template <class E, class F, class S, class T>
                    void operator()(E const&, F& f, S&, T&) const
                    {
                        f.master->ctrlInterfacePtr_.reset(
                            new ProtectionControllerInterface(f.master->robotPtr_));
                    }
            };
            struct ActResetCtrl
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm& f, Src&, Tgt&) const
                    {
                        if (f.master->ctrlInterfacePtr_)
                        {
                            f.master->ctrlInterfacePtr_.reset();
                        }
                    }
            };
            // log
            struct ActLogFsmInit
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation(
                            "FsmRuntimeLoop_FsmNodeType", "INIT");
                    }
            };
            struct ActLogFsmCandidateDefault
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation(
                            "FsmRuntimeLoop_FsmNodeType", "CANDIDATE_DEFAULT");
                    }
            };
            struct ActLogFsmError
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation(
                            "FsmRuntimeLoop_FsmNodeType", "ERROR");
                    }
            };
            struct ActLogFsmProtectionShutdown
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation(
                            "FsmRuntimeLoop_FsmNodeType", "PROTECTION_SHUTDOWN");
                    }
            };
            struct ActLogFsmCalibrationSuccessfully
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation(
                            "FsmRuntimeLoop_FsmNodeType", "EXEC_CALIBRATION_SUCCESSFULLY");
                    }
            };
            struct ActLogFsmCalibrationFailed
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation(
                            "FsmRuntimeLoop_FsmNodeType", "EXEC_CALIBRATION_FAILED");
                    }
            };
            struct ActLogJoyInit
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "INIT");
                    }
            };
            struct ActLogJoyCandidateDefault
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "CANDIDATE_DEFAULT");
                    }
            };
            struct ActLogJoyCandidateTeaching
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "CANDIDATE_TEACHING");
                    }
            };
            struct ActLogJoyCandidateCustom
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "CANDIDATE_CUSTOM");
                    }
            };
            struct ActLogJoyCandidateRemote
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "CANDIDATE_REMOTE");
                    }
            };
            struct ActLogJoyCandidateCalibration
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "CANDIDATE_CALIBRATION");
                    }
            };
            struct ActLogJoyExecDefault
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "EXEC_DEFAULT");
                    }
            };
            struct ActLogJoyExecTeaching
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "EXEC_TEACHING");
                    }
            };
            struct ActLogJoyExecCustom
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "EXEC_COSTOM");
                    }
            };
            struct ActLogJoyExecRemote
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "EXEC_REMOTE");
                    }
            };
            struct ActLogJoyExecCalibration
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "EXEC_CALIBRATING");
                    }
            };
            struct ActLogJoyProtectionShutdown
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    void operator()(Ev const&, Fsm&, Src&, Tgt&) const
                    {
                        livelybot_logger::LoggerInterface::logOperation("JoyMsgCallback_FsmNodeType", "PROTECTION_SHUTDOWN");
                    }
            };
            struct GuardAllNormal
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    bool operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        return fsm.master->allDeviceNormal();
                    }
            };
            struct GuardNotAllNormal
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    bool operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        auto ret = fsm.master->allDeviceNormal();
                        if (!ret)
                        {
                            ROS_ERROR_THROTTLE(1, "fsm %d all not normal", ret);
                        }
                        return !fsm.master->allDeviceNormal();
                    }
            };
            struct GuardShutdown
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    bool operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        auto ptr = fsm.master->ctrlInterfacePtr_;
                        if (ptr && ptr->get_is_shutdowm())
                        {
                            ROS_ERROR_THROTTLE(1, "fsm shutdown");
                        }
                        return ptr && ptr->get_is_shutdowm();
                    }
            };
            // 校验超时 3 秒
            struct GuardTimeout3s
            {
                    template <class Ev, class Fsm, class Src, class Tgt>
                    bool operator()(Ev const&, Fsm& fsm, Src&, Tgt&) const
                    {
                        auto elapsed = ros::Time::now() - fsm.master->stateEntryTime_;
                        ROS_INFO_THROTTLE(1.0, "Calibration in %.0f seconds...", 3.0 - elapsed.toSec());
                        return elapsed.toSec() >= 3.0;
                    }
            };

            using loop_table1 = boost::mpl::vector<
                // loop Guard -> action -> target
                Row<Init, TickEvt, Init, ActSetMotorZero, none>,
                Row<Init, TickEvt, Error, ActLogFsmError, GuardNotAllNormal>,
                Row<Error, TickEvt, Init, ActLogFsmInit, GuardAllNormal>,
                // Row<ProtectionShutdown, TickEvt, Init, boost::mpl::vector<ActResetCtrl, ActLogFsmInit>, GuardShutdown>,
                Row<ProtectionShutdown, TickEvt, Init, ActResetCtrl, GuardShutdown>,

                // candidate
                Row<CandidateDefault, TickEvt, CandidateDefault, ActSendMotor, none>,
                Row<CandidateDefault, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                Row<CandidateCustom, TickEvt, CandidateCustom, ActSendMotor, none>,
                Row<CandidateCustom, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                Row<CandidateRemote, TickEvt, CandidateRemote, ActSendMotor, none>,
                Row<CandidateRemote, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                Row<CandidateCalibration, TickEvt, CandidateCalibration, ActSendMotor, none>,
                Row<CandidateCalibration, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                Row<CandidateTeaching, TickEvt, CandidateTeaching, ActSendMotor, none>,
                Row<CandidateTeaching, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                // exec
                Row<ExecDefault, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardShutdown>,
                Row<ExecDefault, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardNotAllNormal>,
                Row<ExecCustom, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardShutdown>,
                Row<ExecCustom, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardNotAllNormal>,
                Row<ExecRemote, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardShutdown>,
                Row<ExecRemote, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardNotAllNormal>>;

            using loop_table2 = boost::mpl::vector<
                // init
                Row<Init, StartEvt, ExecDefault, ActResetDefaultCtrl, none>,
                // Row<Init, StartEvt, ExecDevelop, ActResetDevelopCtrl, none>,
                // calibration
                Row<ExecCalibrating, TickEvt, ExecCalibrating, ActCheckCalibration, GuardShutdown>,
                Row<ExecCalibrating, CalibrationSuccessEvt, ExecCalibrationSuccess, ActLogFsmCalibrationSuccessfully, none>,
                Row<ExecCalibrating, CalibrationFailedEvt, ExecCalibrationFailed, ActLogFsmCalibrationFailed, none>,
                Row<ExecCalibrationSuccess, TickEvt, CandidateDefault, none, GuardTimeout3s>,
                Row<ExecCalibrationFailed, TickEvt, Error, none, GuardTimeout3s>,

                // teaching
                Row<ExecTeaching, TickEvt, Init, ActLogFsmInit, GuardShutdown>,

                // develop
                Row<CandidateDevelop, TickEvt, CandidateDevelop, ActSendMotor, none>,
                Row<CandidateDevelop, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                Row<ExecDevelop, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardShutdown>,
                Row<ExecDevelop, TickEvt, ProtectionShutdown, ActResetProtectionCtrl, GuardNotAllNormal>,

                Row<Error, TickEvt, Error, ActSendMotor, GuardNotAllNormal>,
                Row<Error, TickEvt, Init, ActSendMotor, GuardAllNormal>>;

            using joy_table1 = boost::mpl::vector<
                // 按键事件转换（JoyMsgCallback 部分）
                Row<Init, LastOptionEvt, CandidateDevelop, ActLogJoyCandidateTeaching, none>,
                Row<Init, NextOptionEvt, CandidateDefault, ActLogJoyCandidateDefault, none>,

                Row<CandidateDefault, ConfirmEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateDefault, LastOptionEvt, Init, ActLogJoyInit, none>,
                Row<CandidateDefault, NextOptionEvt, CandidateCustom, ActLogJoyCandidateCustom, none>,

                Row<CandidateCustom, ConfirmEvt, ExecCustom, ActResetCustomCtrl, none>,
                Row<CandidateCustom, LastOptionEvt, CandidateDefault, ActLogJoyCandidateDefault, none>,
                Row<CandidateCustom, NextOptionEvt, CandidateRemote, ActLogJoyCandidateRemote, none>,

                Row<CandidateRemote, ConfirmEvt, ExecRemote, ActResetRemoteCtrl, none>,
                Row<CandidateRemote, LastOptionEvt, CandidateCustom, ActLogJoyCandidateCustom, none>,
                Row<CandidateRemote, NextOptionEvt, CandidateCalibration, ActLogJoyCandidateCalibration, none>,

                Row<CandidateCalibration, ConfirmEvt, ExecCalibrating, ActResetZeroCalibCtrl, none>,
                Row<CandidateCalibration, LastOptionEvt, CandidateRemote, ActLogJoyCandidateRemote, none>,
                Row<CandidateCalibration, NextOptionEvt, CandidateTeaching, ActLogJoyCandidateTeaching, none>,

                Row<CandidateTeaching, ConfirmEvt, ExecTeaching, ActResetTeachCtrl, none>,
                Row<CandidateTeaching, LastOptionEvt, CandidateCalibration, ActLogJoyCandidateCalibration, none>,
                Row<CandidateTeaching, NextOptionEvt, CandidateDevelop, ActLogJoyInit, none>,

                Row<CandidateDevelop, ConfirmEvt, ExecDevelop, ActResetDevelopCtrl, none>,
                Row<CandidateDevelop, LastOptionEvt, CandidateTeaching, ActLogJoyCandidateTeaching, none>,
                Row<CandidateDevelop, NextOptionEvt, Init, ActLogJoyInit, none>>;

            using joy_table2 = boost::mpl::vector<
                Row<ExecDefault, ProtectionShutdownEvt, ProtectionShutdown, ActResetProtectionCtrl, none>,
                Row<ExecCustom, ProtectionShutdownEvt, ProtectionShutdown, ActResetProtectionCtrl, none>,
                Row<ExecRemote, ProtectionShutdownEvt, ProtectionShutdown, ActResetProtectionCtrl, none>,
                Row<ExecCalibrating, ProtectionShutdownEvt, ProtectionShutdown, ActResetProtectionCtrl, none>,
                Row<ExecTeaching, ProtectionShutdownEvt, ProtectionShutdown, ActResetProtectionCtrl, none>,
                Row<ExecDevelop, ProtectionShutdownEvt, ProtectionShutdown, ActResetProtectionCtrl, none>,
                Row<ProtectionShutdown, ProtectionShutdownEvt, Init, ActResetCtrl, none>>;

            using default_table = boost::mpl::vector<
                Row<Init, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateDefault, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateCustom, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateRemote, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateCalibration, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateTeaching, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<CandidateDevelop, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<ExecCustom, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<ExecRemote, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<ExecCalibrating, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<ExecTeaching, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<ExecDevelop, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>,
                Row<ProtectionShutdown, DefaultModeEvt, ExecDefault, ActResetDefaultCtrl, none>>;

            // using transition_table = boost::mpl::joint_view<boost::mpl::joint_view<loop_table1, loop_table2>, boost::mpl::joint_view<joy_table1, joy_table2>>;
            using transition_table =
                boost::mpl::joint_view<
                    boost::mpl::joint_view<loop_table1, loop_table2>,                     // 第一步：loop1 + loop2
                    boost::mpl::joint_view<default_table,                                 // 第二步：上面的结果 + default_table
                                           boost::mpl::joint_view<joy_table1, joy_table2> // 第三步：上面的结果 + joy1 + joy2
                                           >>;

            // using initial_state = ExecDefault;
            using initial_state = Init;
            // 默认情况下，不处理的事件保持当前状态
            template <class FSM, class Event>
            void no_transition(Event const&, FSM&, int)
            {
                std::cout << "Event not handled, staying in current state." << std::endl;
            }

            template <class Event, class FSM>
            void exception_caught(const Event& evt, FSM&, std::exception& e)
            {
                std::cout << "catch: (Event=" << typeid(Event).name() << ") : " << e.what() << std::endl;
            }

        private:
            FsmNodeType state = FsmNodeType::EXEC_DEFAULT;
    };

};

#endif
