const S = 9;                   // 보드 크기 (모델과 동일)
let board = Array.from({length:S},()=>Array(S).fill(0));
let current = 1;               // 1: 흑, -1: 백

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const CELL = Math.floor(canvas.width / S);

function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // 격자
  for (let i=0;i<S;i++){
    ctx.beginPath();
    ctx.moveTo(CELL/2, CELL/2 + i*CELL);
    ctx.lineTo(CELL/2 + (S-1)*CELL, CELL/2 + i*CELL);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(CELL/2 + i*CELL, CELL/2);
    ctx.lineTo(CELL/2 + i*CELL, CELL/2 + (S-1)*CELL);
    ctx.stroke();
  }
  // 돌
  for (let x=0;x<S;x++) for (let y=0;y<S;y++){
    if (board[x][y] === 0) continue;
    ctx.beginPath();
    ctx.arc(CELL/2 + y*CELL, CELL/2 + x*CELL, CELL*0.4, 0, Math.PI*2);
    ctx.fillStyle = (board[x][y]===1) ? '#000' : '#fff';
    ctx.fill();
    ctx.stroke();
  }
}

canvas.addEventListener('click', async (e)=>{
  const rect = canvas.getBoundingClientRect();
  const y = Math.floor((e.clientX - rect.left) / CELL);
  const x = Math.floor((e.clientY - rect.top) / CELL);
  if (x<0||x>=S||y<0||y>=S) return;
  if (board[x][y] !== 0) return;

  // 사람 착수
  board[x][y] = current;
  current *= -1;
  draw();

  // AI 요청
  const res = await fetch('/ai/api/move/', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ board, current_player: current })
  });
  const data = await res.json();
  if (data.move) {
    const [ax, ay] = data.move;
    board[ax][ay] = current;   // AI 착수
    current *= -1;
    draw();
  } else {
    console.log('게임 종료 또는 둘 곳 없음:', data);
  }
});

draw();
