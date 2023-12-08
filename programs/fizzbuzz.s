  .data
fizzstr:
  .string "Fizz"
buzzstr:
  .string "Buzz"
wrongstr:
  .string "No thanks!"  
  .text
  mov  ax,0x0
  mov  bx,0xf
mainloop:
  add  ax,0x1
  push ax
  call fizzbuzz
  pop  ax
  cmp  ax,bx
  jne  mainloop
  hlt
  nop  
fizzbuzz:
  push bp
  mov  bp,sp
  push 0x0
  push 0x0
  mov  bx,sp
  mov  ax,[bp+0x4]
  push ax
  push bx
  call hex
  pop  bx
  pop  ax
  push 0x2
  push bx
  call print
  pop  bx
  pop  ax
  pop  ax
  mov  ax,[bp+0x4]
  mov  bx,0x3
  mov  dx,0x0
  div  bx
  cmp  dx,0x0
  jne  fizzbuzz_check_five
  pop  ax
  push 0x1
  push 0x4
  push fizzstr
  call print
  pop  ax
  pop  ax
fizzbuzz_check_five:
  mov  ax,[bp+0x4]
  mov  bx,0x5
  mov  dx,0x0
  div  bx
  cmp  dx,0x0
  jne  fizzbuzz_done
  pop  ax
  push 0x1
  push 0x4
  push buzzstr
  call print
  pop  ax
  pop  ax
fizzbuzz_done:
  pop  ax
  cmp  ax,0x1
  je   fizzbuzz_ret
  push 0xa
  push wrongstr
  call print
  pop  ax
  pop  ax
fizzbuzz_ret:
  push 0xa
  mov  bx,sp
  push 0x1
  push bx
  call print
  pop  ax
  pop  ax
  pop  ax
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
