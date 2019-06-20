fo=open('bg.txt','r+')    
lines=fo.read() 
fo.close()  
fo=open('bg2.txt','w')     
for i in lines.split('\n')[:-1]: 
    fo.write('/root/opencv_workspace/'+i+ '\n') 
fo.close()
