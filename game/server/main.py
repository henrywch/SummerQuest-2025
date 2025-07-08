from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import random
import json
from typing import Dict, List, Optional
from enum import Enum
import asyncio

# 导入卡牌数据
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cards'))
from v0 import CARDS_V0
from v1 import CARDS_V1
from __base__ import Card, CardType

app = FastAPI(title="成语卡牌对战游戏服务器")

# 静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

class GameState(Enum):
    WAITING = "waiting"  # 等待玩家加入
    READY = "ready"      # 双方已加入，等待开始
    PLAYING = "playing"  # 游戏进行中
    FINISHED = "finished" # 游戏结束

class Player:
    def __init__(self, player_id: str, websocket: WebSocket = None):
        self.id = player_id
        self.websocket = websocket
        self.hand_cards: List[Card] = []  # 手牌
        self.score_cards: List[Card] = []  # 得分区
        self.ready = False
    
    def to_dict(self, show_hand_cards=True):
        """转换为字典格式
        
        Args:
            show_hand_cards: 是否显示手牌详细信息，False时只显示手牌数量
        """
        result = {
            "id": self.id,
            "hand_count": len(self.hand_cards),
            "score_count": len(self.score_cards),
            "score_cards": [{
                "id": card.id,
                "name": card.name,
                "card_type": card.card_type.value
            } for card in self.score_cards],
            "ready": self.ready
        }
        
        # 只有在允许的情况下才显示手牌详细信息
        if show_hand_cards:
            result["hand_cards"] = [{
                "id": card.id,
                "name": card.name,
                "card_type": card.card_type.value,
                "effect_description": card.effect_description
            } for card in self.hand_cards]
        else:
            result["hand_cards"] = []  # 不显示手牌详细信息
            
        return result

class GameRoom:
    def __init__(self, room_id: str):
        self.id = room_id
        self.state = GameState.WAITING
        self.players: Dict[str, Player] = {}  # key -> Player
        self.deck: List[Card] = []  # 牌库
        self.discard_pile: List[Card] = []  # 弃牌区
        self.current_turn = 0  # 当前回合玩家索引
        self.turn_count = 0  # 回合计数
        
    def add_player(self, key: str, websocket: WebSocket) -> bool:
        """添加玩家到房间"""
        if len(self.players) >= 2:
            return False
        
        player = Player(key, websocket)
        self.players[key] = player
        
        if len(self.players) == 2:
            self.state = GameState.READY
        
        return True
    
    def remove_player(self, key: str):
        """移除玩家"""
        if key in self.players:
            del self.players[key]
        
        if len(self.players) < 2:
            self.state = GameState.WAITING
    
    def start_game(self):
        """开始游戏"""
        if self.state != GameState.READY or len(self.players) != 2:
            return False
        
        # 初始化牌库：v0 + v1 共40张卡
        all_cards = CARDS_V0 + CARDS_V1
        self.deck = all_cards.copy()
        random.shuffle(self.deck)
        
        # 发牌：每人5张手牌
        player_keys = list(self.players.keys())
        for i in range(5):
            for key in player_keys:
                if self.deck:
                    card = self.deck.pop(0)
                    self.players[key].hand_cards.append(card)
        
        self.state = GameState.PLAYING
        self.current_turn = 0
        return True
    
    def get_current_player_key(self) -> Optional[str]:
        """获取当前回合玩家的key"""
        if len(self.players) != 2:
            return None
        player_keys = list(self.players.keys())
        return player_keys[self.current_turn]
    
    def to_dict(self, requesting_player_key=None):
        """转换为字典格式用于前端展示
        
        Args:
            requesting_player_key: 请求游戏状态的玩家key，只有该玩家能看到自己的手牌详情
                                  如果为None，则表示观战者，可以看到所有玩家的手牌
        """
        player_keys = list(self.players.keys())
        return {
            "room_id": self.id,
            "state": self.state.value,
            "deck_count": len(self.deck),
            "discard_count": len(self.discard_pile),
            "current_turn": self.current_turn,
            "turn_count": self.turn_count,
            "players": {
                key: self.players[key].to_dict(show_hand_cards=(key == requesting_player_key or requesting_player_key is None))
                for key in player_keys
            },
            "current_player": self.get_current_player_key()
        }

# 全局游戏房间管理
game_rooms: Dict[str, GameRoom] = {}
connected_clients: Dict[str, WebSocket] = {}  # room_id -> websocket (观战者)

# API模型
class JoinRoomRequest(BaseModel):
    pass

class StartGameRequest(BaseModel):
    key: str

@app.get("/", response_class=HTMLResponse)
async def get_game_page():
    """游戏主页面"""
    return HTMLResponse(content=open("templates/index.html", "r", encoding="utf-8").read())

@app.post("/api/create_room")
async def create_room():
    """创建游戏房间"""
    room_id = str(uuid.uuid4())[:8]
    game_rooms[room_id] = GameRoom(room_id)
    
    return {
        "success": True,
        "room_id": room_id,
        "message": "房间创建成功"
    }

@app.post("/api/join_room/{room_id}")
async def join_room(room_id: str, request: JoinRoomRequest):
    """加入游戏房间"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if len(room.players) >= 2:
        raise HTTPException(status_code=400, detail="房间已满")
    
    # 生成玩家key
    player_key = str(uuid.uuid4())[:12]
    
    # 创建玩家并添加到房间
    player = Player(player_key)
    room.players[player_key] = player
    
    if len(room.players) == 2:
        room.state = GameState.READY
    
    # 广播状态更新
    await broadcast_game_state(room_id)
    
    return {
        "success": True,
        "key": player_key,
        "room_id": room_id,
        "player_count": len(room.players),
        "message": "加入房间成功"
    }

@app.post("/api/start_game/{room_id}")
async def start_game(room_id: str, request: StartGameRequest):
    """开始游戏"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if request.key not in room.players:
        raise HTTPException(status_code=403, detail="无效的玩家key")
    
    # 检查房间人数
    if len(room.players) < 2:
        raise HTTPException(status_code=400, detail=f"房间人数不足，当前{len(room.players)}/2人")
    
    if room.state != GameState.READY:
        raise HTTPException(status_code=400, detail=f"游戏状态不正确，当前状态：{room.state.value}")
    
    # 标记玩家准备
    room.players[request.key].ready = True
    
    # 广播状态更新
    await broadcast_game_state(room_id)
    
    # 检查是否所有玩家都准备好了
    all_ready = all(player.ready for player in room.players.values())
    
    if all_ready:
        success = room.start_game()
        if success:
            # 广播游戏开始
            await broadcast_game_state(room_id)
            return {
                "success": True,
                "message": "游戏开始！",
                "game_state": room.to_dict(requesting_player_key=request.key)
            }
        else:
            raise HTTPException(status_code=500, detail="游戏启动失败")
    else:
        return {
            "success": True,
            "message": "等待其他玩家准备",
            "ready_count": sum(1 for p in room.players.values() if p.ready)
        }

@app.get("/api/game_state/{room_id}")
async def get_game_state(room_id: str, player_key: str = None):
    """获取游戏状态
    
    Args:
        room_id: 房间ID
        player_key: 玩家key，用于确定显示哪些手牌信息
    """
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    return {
        "success": True,
        "game_state": room.to_dict(requesting_player_key=player_key)
    }

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, player_key: str = None):
    """WebSocket连接用于实时更新"""
    await websocket.accept()
    
    if room_id not in game_rooms:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "房间不存在"
        }))
        await websocket.close()
        return
    
    # 添加到连接列表，包含玩家key信息
    connection_key = f"{room_id}_{id(websocket)}"
    connected_clients[connection_key] = {
        "websocket": websocket,
        "player_key": player_key
    }
    
    try:
        # 发送当前游戏状态
        room = game_rooms[room_id]
        await websocket.send_text(json.dumps({
            "type": "game_state",
            "data": room.to_dict(requesting_player_key=player_key)
        }))
        
        # 保持连接
        while True:
            data = await websocket.receive_text()
            # 这里可以处理客户端发送的消息
            
    except WebSocketDisconnect:
        # 移除连接
        if connection_key in connected_clients:
            del connected_clients[connection_key]

async def broadcast_game_state(room_id: str):
    """广播游戏状态给所有连接的客户端"""
    if room_id not in game_rooms:
        return
    
    room = game_rooms[room_id]
    
    # 发送给所有相关的websocket连接
    disconnected = []
    for key, client_info in connected_clients.items():
        if key.startswith(room_id):
            try:
                # 为每个客户端生成个性化的游戏状态
                websocket = client_info["websocket"]
                player_key = client_info["player_key"]
                
                message = json.dumps({
                    "type": "game_state",
                    "data": room.to_dict(requesting_player_key=player_key)
                })
                
                await websocket.send_text(message)
            except:
                disconnected.append(key)
    
    # 清理断开的连接
    for key in disconnected:
        del connected_clients[key]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)