TARGET_BIN=$1
xtensa-elf-objcopy -O binary  -j .clib.rodata  -j .rodata  -j .clib.data -j .rtos.percpu.data -j .data -j .bss --set-section-flags .bss=alloc,load,contents --gap-fill 0x00 "$TARGET_BIN" DRAM0.bin
xtensa-elf-objcopy -O binary  -j .dram1.rodata  -j .dram1.data  --gap-fill 0x00 "$TARGET_BIN" DRAM1.bin 
xtensa-elf-objcopy -O binary  -j *.literal -j *.text   --gap-fill 0x00 "$TARGET_BIN" IRAM0.bin 
xtensa-elf-objcopy -O binary -j .iram1.*   --gap-fill 0x00  "$TARGET_BIN"  IRAM1.bin 
xtensa-elf-objcopy -O binary -j .drom0.* -j .drom.rodata  --gap-fill 0x00  "$TARGET_BIN" DROM0.bin 
xtensa-elf-objcopy -O binary -j .irom*   --gap-fill 0x00 "$TARGET_BIN" IROM0.bin 
xtensa-elf-objcopy -O binary -j .sram0*   --gap-fill 0x00 "$TARGET_BIN" SRAM.bin 
