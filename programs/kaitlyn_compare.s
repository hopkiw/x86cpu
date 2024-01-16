  mov   ax,0x1
  push  ax
  pop   ax
  sub   ax,0x3
  je    success
failure:
  mov   ax,0x2  ; failure: the number was not 3
  jmp   done
success:
  mov   ax,0x1  ; success: the number was 3
done:
  hlt
