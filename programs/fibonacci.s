  .data
prefix:
  .string "0x"
  .text
  call fibonacci
  hlt
  nop  
fibonacci:
  push bp
  mov  bp,sp
  sub  sp,0x6
  mov  [bp-0x2],0x1  ; int a
  mov  [bp-0x4],0x2  ; int b
fibonacci_loop:
  mov  ax,[bp-0x2]
  mov  bx,[bp-0x4]
  add  ax,[bp-0x4]
  mov  [bp-0x2],bx
  mov  [bp-0x4],ax
  ; printing
  push 0x2
  push prefix
  call print
  add  sp,0x4
  push [bp-0x4]
  push bp
  sub  [sp],0x6
  call hex
  pop  bx
  add  sp,0x2
  push 0x2
  push bx
  call print
  pop  bx
  add  sp,0x2
  mov  [bp-0x6],0x000a
  push 0x1
  push bx
  call print
  add  sp,0x4
  jmp  fibonacci_loop
  add  sp,0x6
  pop  bp
  ret
print:
  push bp
  mov  bp,sp
  mov  bx,[bp+0x4]
  mov  cx,[bp+0x6]
  mov  dx,0x0
print_loop:
  mov  ax,[bx+dx]
  out  0x20,al
  add  dx,0x1
  cmp  cx,dx
  jne  print_loop
  pop  bp
  ret
  hlt
hex:
  push  bp
  mov   bp,sp
  mov   bx,[bp+0x4]
  mov   ax,[bp+0x6]
  mov   cx,0x10
  mov   dx,0x0
  div   cx
  cmp   ax,0xa
  jg    hex_addalpha
  add   ax,0x30
  jmp   hex_dx
hex_addalpha:
  add   ax,0x57
hex_dx:
  cmp   dx,0xa
  jg    hex_addalpha2
  add   dx,0x30
  jmp   hex_push
hex_addalpha2:
  add   dx,0x57
hex_push:
  mov   [bx],ax
  or    [bx+0x1],dx
  pop   bp
  ret
  hlt
