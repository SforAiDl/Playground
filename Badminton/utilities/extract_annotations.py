from xml.etree import ElementTree
import os
import cv2
def extract_annotation(file_path,video_path,save_loc='./Badminton/dataset/'):
        if(save_loc[-1] != '/'):
            save_loc = save_loc+'/'
        if not os.path.exists(save_loc + 'annots'):
            os.makedirs(save_loc + 'annots')
            
        if not os.path.exists(save_loc+'annots/n'):
                os.makedirs(save_loc+'annots/n')
                
        if not os.path.exists(save_loc+'annots/lbpb'):
                os.makedirs(save_loc+'annots/lbpb')
        if not os.path.exists(save_loc+'annots/lbpt'):
                os.makedirs(save_loc+'annots/lbpt')
                
        if not os.path.exists(save_loc+'annots/bhpb'):
                os.makedirs(save_loc+'annots/bhpb')
        if not os.path.exists(save_loc+'annots/bhpt'):
                os.makedirs(save_loc+'annots/bhpt')
                
        if not os.path.exists(save_loc+'annots/fhpb'):
                os.makedirs(save_loc+'annots/fhpb')
        if not os.path.exists(save_loc+'annots/fhpt'):
                os.makedirs(save_loc+'annots/fhpt')
                
        if not os.path.exists(save_loc+'annots/smpb'):
                os.makedirs(save_loc+'annots/smpb')
        if not os.path.exists(save_loc+'annots/smpt'):
                os.makedirs(save_loc+'annots/smpt')
                
        if not os.path.exists(save_loc+'annots/spb'):
                os.makedirs(save_loc+'annots/spb')
        if not os.path.exists(save_loc+'annots/spt'):
                os.makedirs(save_loc+'annots/spt')
                
        if not os.path.exists(save_loc+'annots/rtpb'):
                os.makedirs(save_loc+'annots/rtpb')
        if not os.path.exists(save_loc+'annots/rtpt'):
                os.makedirs(save_loc+'annots/rtpt')

        tree = ElementTree.parse(file_path)
        root = tree.getroot()
        dic={}
        Tier = root[3]
        Time = root[1]
        vidcap = cv2.VideoCapture(video_path)

        for i in Time:
            dic[i.attrib['TIME_SLOT_ID']] = int(i.attrib['TIME_VALUE'])
            
        for i in Tier:
            for j in i:
                for k in j:
                    for m in range(dic[j.attrib['TIME_SLOT_REF1']], dic[j.attrib['TIME_SLOT_REF2']],40):
                            vidcap.set(cv2.CAP_PROP_POS_MSEC,m)      
                            success,image = vidcap.read()
                            if success:
                                if not os.path.exists(save_loc+'annots/'+k.text+'/'+str(dic[j.attrib['TIME_SLOT_REF1']])):
                                    os.makedirs(save_loc+'annots/'+k.text+'/'+str(dic[j.attrib['TIME_SLOT_REF1']]))
                                cv2.imwrite(save_loc+"./annots/"+k.text+"/"+str(dic[j.attrib['TIME_SLOT_REF1']])+"/frame" +str(m)+".jpg", image)     # save frame as JPEG file                                            
                                
                                
                                
                                
