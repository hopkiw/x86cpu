  mov  ax,0x7e14
  push ax
  mov  ax,0x30
  push ax
  mov  ax,0xDEAF
  push ax
  mov  ax,0x0
  ; start of program 
  pop  cx
  pop  bx
  add  bx,0x1
  pop  ax
loop:
  mov  [ax],cx
  add  ax,0x2
  sub  bx,0x1
  je   done
  jmp  loop
done:
  hlt
  
