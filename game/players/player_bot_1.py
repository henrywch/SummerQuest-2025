#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
玩家模拟程序 1
自动连接到指定房间，执行确认和出牌操作
"""

import asyncio
import aiohttp
import websockets
import json
import sys
from typing import Optional

class PlayerBot:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.ws_url = server_url.replace("http", "ws")
        self.room_id: Optional[str] = None
        self.player_key: Optional[str] = None
        self.websocket = None
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def join_room(self, room_id: str) -> bool:
        """加入指定房间"""
        try:
            url = f"{self.server_url}/api/join_room/{room_id}"
            async with self.session.post(url, json={}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        self.room_id = room_id
                        self.player_key = data.get("key")
                        print(f"[玩家Bot1] 成功加入房间 {room_id}，玩家key: {self.player_key}")
                        return True
                    else:
                        print(f"[玩家Bot1] 加入房间失败: {data.get('message')}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"[玩家Bot1] 加入房间失败，HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            print(f"[玩家Bot1] 加入房间异常: {e}")
            return False
            
    async def start_game(self) -> bool:
        """确认开始游戏"""
        if not self.room_id or not self.player_key:
            print("[玩家Bot1] 尚未加入房间")
            return False
            
        try:
            url = f"{self.server_url}/api/start_game/{self.room_id}"
            payload = {"key": self.player_key}
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        print(f"[玩家Bot1] 确认开始游戏: {data.get('message')}")
                        return True
                    else:
                        print(f"[玩家Bot1] 开始游戏失败: {data.get('message')}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"[玩家Bot1] 开始游戏失败，HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            print(f"[玩家Bot1] 开始游戏异常: {e}")
            return False
            
    async def get_game_state(self) -> Optional[dict]:
        """获取游戏状态"""
        if not self.room_id:
            return None
            
        try:
            url = f"{self.server_url}/api/game_state/{self.room_id}"
            params = {"player_key": self.player_key} if self.player_key else {}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("game_state")
                else:
                    print(f"[玩家Bot1] 获取游戏状态失败，HTTP {response.status}")
                    return None
        except Exception as e:
            print(f"[玩家Bot1] 获取游戏状态异常: {e}")
            return None
            
    async def connect_websocket(self):
        """连接WebSocket接收实时更新"""
        if not self.room_id:
            print("[玩家Bot1] 尚未加入房间，无法连接WebSocket")
            return
            
        try:
            ws_url = f"{self.ws_url}/ws/{self.room_id}"
            if self.player_key:
                ws_url += f"?player_key={self.player_key}"
                
            async with websockets.connect(ws_url) as websocket:
                self.websocket = websocket
                print(f"[玩家Bot1] WebSocket连接成功")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_websocket_message(data)
                    except json.JSONDecodeError:
                        print(f"[玩家Bot1] 收到无效JSON消息: {message}")
                        
        except Exception as e:
            print(f"[玩家Bot1] WebSocket连接异常: {e}")
            
    async def handle_websocket_message(self, data: dict):
        """处理WebSocket消息"""
        msg_type = data.get("type")
        
        if msg_type == "game_state":
            game_data = data.get("data", {})
            state = game_data.get("state")
            current_player = game_data.get("current_player")
            
            print(f"[玩家Bot1] 游戏状态更新: {state}")
            
            if state == "playing" and current_player == self.player_key:
                print(f"[玩家Bot1] 轮到我出牌了！")
                await self.play_card(game_data)
                
        elif msg_type == "error":
            print(f"[玩家Bot1] 收到错误消息: {data.get('message')}")
            
    async def play_card(self, game_state: dict):
        """出牌逻辑（目前只是模拟，因为服务器还没有出牌API）"""
        players = game_state.get("players", {})
        my_data = players.get(self.player_key, {})
        hand_cards = my_data.get("hand_cards", [])
        
        if hand_cards:
            # 简单策略：出第一张牌
            card_to_play = hand_cards[0]
            print(f"[玩家Bot1] 准备出牌: {card_to_play.get('name')} (ID: {card_to_play.get('id')})")
            
            # TODO: 这里需要调用出牌API，但目前服务器还没有实现
            print(f"[玩家Bot1] 注意：服务器尚未实现出牌API，无法实际出牌")
        else:
            print(f"[玩家Bot1] 手牌为空，无法出牌")
            
    async def run(self, room_id: str):
        """运行机器人"""
        print(f"[玩家Bot1] 启动，准备加入房间: {room_id}")
        
        # 加入房间
        if not await self.join_room(room_id):
            return
            
        # 等待房间状态变为READY（两个玩家都加入）
        print(f"[玩家Bot1] 等待其他玩家加入...")
        max_wait_time = 30  # 最多等待30秒
        wait_time = 0
        
        while wait_time < max_wait_time:
            game_state = await self.get_game_state()
            if game_state and game_state.get("state") == "ready":
                print(f"[玩家Bot1] 房间已准备就绪，开始游戏确认")
                break
            await asyncio.sleep(1)
            wait_time += 1
            
        if wait_time >= max_wait_time:
            print(f"[玩家Bot1] 等待超时，退出")
            return
            
        # 等待额外2秒确保两个机器人都准备好
        await asyncio.sleep(2)
        await self.start_game()
        
        # 连接WebSocket监听游戏状态
        await self.connect_websocket()

async def main():
    if len(sys.argv) != 2:
        print("使用方法: python player_bot_1.py <房间ID>")
        sys.exit(1)
        
    room_id = sys.argv[1]
    
    async with PlayerBot() as bot:
        await bot.run(room_id)

if __name__ == "__main__":
    asyncio.run(main())