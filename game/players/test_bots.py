#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人测试脚本
自动创建房间并启动两个机器人进行测试
"""

import asyncio
import aiohttp
import subprocess
import sys
import time
from typing import Optional

class BotTester:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.room_id: Optional[str] = None
        
    async def create_room(self) -> Optional[str]:
        """创建游戏房间"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/api/create_room") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success"):
                            room_id = data.get("room_id")
                            print(f"✅ 成功创建房间: {room_id}")
                            return room_id
                        else:
                            print(f"❌ 创建房间失败: {data.get('message')}")
                            return None
                    else:
                        print(f"❌ 创建房间失败，HTTP {response.status}")
                        return None
        except Exception as e:
            print(f"❌ 创建房间异常: {e}")
            return None
            
    def start_bot(self, bot_script: str, room_id: str) -> subprocess.Popen:
        """启动机器人进程"""
        try:
            cmd = [sys.executable, bot_script, room_id]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"🤖 启动机器人: {bot_script} (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"❌ 启动机器人失败: {e}")
            return None
            
    def monitor_bot_output(self, process: subprocess.Popen, bot_name: str, max_lines: int = 20):
        """监控机器人输出"""
        lines_read = 0
        try:
            for line in process.stdout:
                if lines_read >= max_lines:
                    break
                print(f"{bot_name}: {line.strip()}")
                lines_read += 1
                
                # 检查进程是否结束
                if process.poll() is not None:
                    break
                    
        except Exception as e:
            print(f"❌ 监控{bot_name}输出异常: {e}")
            
    async def run_test(self):
        """运行完整测试"""
        print("🚀 开始机器人测试...")
        print("="*50)
        
        # 1. 创建房间
        print("📝 步骤1: 创建游戏房间")
        room_id = await self.create_room()
        if not room_id:
            print("❌ 测试失败：无法创建房间")
            return
            
        print(f"🏠 房间ID: {room_id}")
        print("="*50)
        
        # 2. 启动机器人
        print("📝 步骤2: 启动机器人")
        
        bot1_process = self.start_bot("player_bot_1.py", room_id)
        if not bot1_process:
            print("❌ 测试失败：无法启动Bot1")
            return
            
        # 等待一秒再启动第二个机器人
        await asyncio.sleep(1)
        
        bot2_process = self.start_bot("player_bot_2.py", room_id)
        if not bot2_process:
            print("❌ 测试失败：无法启动Bot2")
            bot1_process.terminate()
            return
            
        print("="*50)
        
        # 3. 监控输出
        print("📝 步骤3: 监控机器人运行（显示前20行输出）")
        print("💡 提示：机器人将自动加入房间、确认开始游戏并尝试出牌")
        print("⚠️  注意：由于服务器尚未实现出牌API，机器人只能模拟出牌")
        print("="*50)
        
        # 并发监控两个机器人的输出
        await asyncio.gather(
            asyncio.create_task(asyncio.to_thread(
                self.monitor_bot_output, bot1_process, "Bot1", 20
            )),
            asyncio.create_task(asyncio.to_thread(
                self.monitor_bot_output, bot2_process, "Bot2", 20
            ))
        )
        
        print("="*50)
        print("📝 步骤4: 清理进程")
        
        # 4. 清理
        try:
            bot1_process.terminate()
            bot2_process.terminate()
            
            # 等待进程结束
            bot1_process.wait(timeout=5)
            bot2_process.wait(timeout=5)
            
            print("✅ 机器人进程已清理")
        except subprocess.TimeoutExpired:
            print("⚠️  强制终止机器人进程")
            bot1_process.kill()
            bot2_process.kill()
        except Exception as e:
            print(f"❌ 清理进程异常: {e}")
            
        print("="*50)
        print("🎉 测试完成！")
        print(f"🌐 你可以访问 {self.server_url} 查看Web界面")
        print(f"🏠 测试房间ID: {room_id}")
        
    async def run_test_with_room_id(self, room_id: str):
        """使用指定房间ID运行测试"""
        print("🚀 开始机器人测试...")
        print("="*50)
        
        # 验证房间是否存在
        print("📝 步骤1: 验证房间状态")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/api/game_state/{room_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        game_state = data.get("game_state", {})
                        state = game_state.get("state")
                        player_count = len(game_state.get("players", {}))
                        print(f"✅ 房间存在，当前状态: {state}，玩家数量: {player_count}")
                        
                        if player_count >= 2:
                            print("⚠️  警告：房间已有2名玩家，机器人可能无法加入")
                    else:
                        print(f"❌ 房间不存在或无法访问，HTTP {response.status}")
                        return
        except Exception as e:
            print(f"❌ 验证房间异常: {e}")
            return
            
        print(f"🏠 房间ID: {room_id}")
        print("="*50)
        
        # 启动机器人
        print("📝 步骤2: 启动机器人")
        
        bot1_process = self.start_bot("player_bot_1.py", room_id)
        if not bot1_process:
            print("❌ 测试失败：无法启动Bot1")
            return
            
        # 等待一秒再启动第二个机器人
        await asyncio.sleep(1)
        
        bot2_process = self.start_bot("player_bot_2.py", room_id)
        if not bot2_process:
            print("❌ 测试失败：无法启动Bot2")
            bot1_process.terminate()
            return
            
        print("="*50)
        
        # 监控输出
        print("📝 步骤3: 监控机器人运行（显示前20行输出）")
        print("💡 提示：机器人将自动加入房间、确认开始游戏并尝试出牌")
        print("⚠️  注意：由于服务器尚未实现出牌API，机器人只能模拟出牌")
        print("="*50)
        
        # 并发监控两个机器人的输出
        await asyncio.gather(
            asyncio.create_task(asyncio.to_thread(
                self.monitor_bot_output, bot1_process, "Bot1", 20
            )),
            asyncio.create_task(asyncio.to_thread(
                self.monitor_bot_output, bot2_process, "Bot2", 20
            ))
        )
        
        print("="*50)
        print("📝 步骤4: 清理进程")
        
        # 清理
        try:
            bot1_process.terminate()
            bot2_process.terminate()
            
            # 等待进程结束
            bot1_process.wait(timeout=5)
            bot2_process.wait(timeout=5)
            
            print("✅ 机器人进程已清理")
        except subprocess.TimeoutExpired:
            print("⚠️  强制终止机器人进程")
            bot1_process.kill()
            bot2_process.kill()
        except Exception as e:
            print(f"❌ 清理进程异常: {e}")
            
        print("="*50)
        print("🎉 测试完成！")
        print(f"🌐 你可以访问 {self.server_url} 查看Web界面")
        print(f"🏠 测试房间ID: {room_id}")
        
async def main():
    print("🤖 机器人测试工具")
    print("本工具将使用指定房间ID启动两个机器人进行测试")
    print()
    
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python test_bots.py <房间ID>")
        print("示例: python test_bots.py abcd1234")
        print()
        print("💡 提示：请先通过Web界面或API创建房间，然后使用房间ID运行此脚本")
        sys.exit(1)
        
    room_id = sys.argv[1]
    print(f"🏠 使用房间ID: {room_id}")
    print()
    
    # 检查机器人文件是否存在
    import os
    bot_files = ["player_bot_1.py", "player_bot_2.py"]
    missing_files = [f for f in bot_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 缺少机器人文件: {', '.join(missing_files)}")
        print("请确保在正确的目录中运行此脚本")
        return
        
    # 检查依赖
    try:
        import aiohttp
        import websockets
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r bot_requirements.txt")
        return
        
    tester = BotTester()
    await tester.run_test_with_room_id(room_id)

if __name__ == "__main__":
    asyncio.run(main())