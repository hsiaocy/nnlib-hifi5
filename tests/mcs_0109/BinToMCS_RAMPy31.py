# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 20:27:05 2016

@author: yc.ting
"""
import sys
#Name='./bin/sys_dram'
Name=sys.argv[1]

DualBank = int(sys.argv[3])
LineByteNum = 16*1  #use 8 for HiFi Mini and HiFi4 DRAM, use 16 for HiFi4 IRAM

#Fidx = 0
BlockNum = 1
#BlockSize = int(32*1024/LineByteNum)
#2025.11.27 larry
MEM_SIZE_K=int(sys.argv[2])
BlockSize = int(MEM_SIZE_K*1024/LineByteNum)

BlockBound = int(BlockSize*LineByteNum)
#LastBlock = int(32*1024/LineByteNum)
#2025.11.27 larry
LastBlock = int(MEM_SIZE_K*1024/LineByteNum)
HalfBlock = int(BlockSize/2)
InFile = open(Name+'.bin','rb')

InData = InFile.read()
InFile.close()
ByteNum=len(InData)

ZeroPadding = 0*1024/LineByteNum

if DualBank == 0:

    for Fidx in range(0,BlockNum):
        OutFile = open(Name+'_%d.mcs'%Fidx,'w')
        
        if ZeroPadding!=0:
            for i in range(0,ZeroPadding):
                OutFile.write('0000000000000000\n')
            ZeroPadding=0
        
        if Fidx == (BlockNum-1):
            BlockSize=LastBlock
            if ByteNum>(LastBlock*LineByteNum):
                print ('Bin size overflow!')
        print (ByteNum)
        
        for i in range(0,BlockSize):
            if ByteNum>LineByteNum:
                ByteNum=ByteNum-LineByteNum
                for j in range(LineByteNum-1,-1,-1):
                    s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum+j])
                    OutFile.write(s1)
            elif ByteNum>0:
                Res=LineByteNum-ByteNum
                for j in range(0,Res):
                    OutFile.write('FF')
                #for j in range(0,ByteNum):
                for j in range(ByteNum-1,-1,-1):
                    s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum+j])
                    OutFile.write(s1)
                ByteNum=0
            else:
                for j in range(0,LineByteNum):
                    OutFile.write('FF')
        
            OutFile.write('\n')
            #OutFile.write('\r\n')
        OutFile.close()
else:
    print ('Dual Bank')
    
    if (ZeroPadding%2) == 1:
        print ('Zero Padding must be even number!')
    
    if (BlockSize%16) == 1:
        print ('Block size must be even words!')
    
    for Fidx in range(0,BlockNum):
        OutFile1 = open(Name+'Low_%d.mcs'%Fidx,'w')
        OutFile2 = open(Name+'High_%d.mcs'%Fidx,'w')
        
        if ZeroPadding!=0:
            for i in range(0,ZeroPadding/2):
                for j in range(0,LineByteNum):
                    OutFile1.write('00')
                    OutFile2.write('00')
                OutFile1.write('\n')
                OutFile2.write('\n')
            ZeroPadding=0
       
        if Fidx == (BlockNum-1):
            BlockSize=LastBlock
            if ByteNum>(LastBlock*LineByteNum):
                print ('Bin size overflow!')
        print (ByteNum)
        
        for i in range(0,HalfBlock):
            if ByteNum>(LineByteNum*2):
                ByteNum=ByteNum-LineByteNum*2
                for j in range(LineByteNum-1,-1,-1):
                    s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum*2+j])
                    OutFile1.write(s1)
                for j in range(LineByteNum*2-1,LineByteNum-1,-1):
                    s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum*2+j])
                    OutFile2.write(s1)
            elif ByteNum>0:
                
                if ByteNum>LineByteNum:
                    Res=LineByteNum*2-ByteNum
                    for j in range(LineByteNum-1,-1,-1):
                        s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum*2+j])
                        OutFile1.write(s1)
                    for j in range(0,Res):
                        OutFile2.write('FF')
                    for j in range(ByteNum-LineByteNum-1,-1,-1):
                        s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum*2+j])
                        OutFile2.write(s1)
                else:
                    Res=LineByteNum-ByteNum
                    for j in range(0,Res):
                        OutFile1.write('FF')
                    #for j in range(0,ByteNum):
                    for j in range(ByteNum-1,-1,-1):
                        s1 = '%02X'%(InData[BlockBound*Fidx+i*LineByteNum*2+j])
                        OutFile1.write(s1)
                    for j in range(0,LineByteNum):
                        OutFile2.write('FF')
                ByteNum=0
            else:
                for j in range(0,LineByteNum):
                    OutFile1.write('FF')
                    OutFile2.write('FF')
        
            OutFile1.write('\n')
            OutFile2.write('\n')
            #OutFile.write('\r\n')
        OutFile1.close()
        OutFile2.close()
    