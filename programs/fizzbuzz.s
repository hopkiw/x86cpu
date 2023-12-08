  .data
fizzstr:
  .string "Fizz"
buzzstr:
  .string "Buzz"
wrongstr:
  .string "No thanks!"  
  .text
  mov  ax,0xb
  push 0xf
mainloop:
  add  ax,0x1
  push ax
  call fizzbuzz
  pop  ax
  mov  bx,[sp]
  cmp  ax,bx
  jne  mainloop
  hlt
  nop  
fizzbuzz:
  push bp
  mov  bp,sp
  sub  sp,0x6
  mov  [bp-0x2],0x0
  ; mov [bp-0x4],0x0  ; redundant, but space for pointer
  push [bp+0x4]
  push bp
  sub  [sp],0x4
  call hex
  pop  bx
  add  sp,0x2
  push 0x2
  push bx
  call print
  pop  bx
  add  sp,0x2
  mov  [bp-0x4],0x203a
  push 0x2
  push bx
  call print
  add  sp,0x4
fizzbuzz_check_three:
  mov  ax,[bp+0x4]
  mov  bx,0x3
  mov  dx,0x0
  div  bx
  cmp  dx,0x0
  jne  fizzbuzz_check_five
  mov  [bp-0x4],0x1
  push 0x4
  push fizzstr
  call print
  add  sp,0x4
fizzbuzz_check_five:
  mov  ax,[bp+0x4]
  mov  bx,0x5
  mov  dx,0x0
  div  bx
  cmp  dx,0x0
  jne  fizzbuzz_done
  mov  [bp-0x4],0x1
  push 0x4
  push buzzstr
  call print
  add  sp,0x4
fizzbuzz_done:
  cmp  [bp-0x4],0x1
  je   fizzbuzz_ret
  push 0xa
  push wrongstr
  call print
  add  sp,0x4
fizzbuzz_ret:
  push 0xa
  mov  bx,sp
  push 0x1
  push bx
  call print
  add  sp,0xc
  pop  bp
  ret
  hlt
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
  mov   [bx+0x1],dx
  pop   bp
  ret
  hlt
