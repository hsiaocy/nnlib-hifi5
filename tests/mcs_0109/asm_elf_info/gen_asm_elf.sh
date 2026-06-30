readelf -a xa_kws_ref_model_test >>xa_kws_ref_model_test.elf_info3
readelf -S --wide xa_kws_ref_model_test >>xa_kws_ref_model_test.elf_info4
xt-objdump -d xa_kws_ref_model_test >>xa_kws_ref_model_test.asm2
