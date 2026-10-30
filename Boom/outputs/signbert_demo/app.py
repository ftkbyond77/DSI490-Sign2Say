from pathlib import Path
import json,tempfile
import numpy as np
import torch
import streamlit as st
import sys
ROOT=Path(__file__).parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model import HandTransformer
st.set_page_config(page_title='ภาษามือไทย • SignBERT-inspired',page_icon='🤟',layout='wide')
st.title('รู้จำคำภาษามือไทย')
st.caption('SignBERT-inspired • Transformer + masked keypoint reconstruction • ไม่ใช้ MANO')
checkpoint=ROOT/'runs/finetune.pt'
if not checkpoint.exists():
    st.warning('ยังไม่มีโมเดลที่ฝึกแล้ว ต้องสกัด keypoints และรัน pretraining / fine-tuning ก่อน')
    st.code('python prepare.py\npython train.py pretrain\npython train.py finetune',language='bash')
    st.stop()

@st.cache_resource
def load_model(path,mtime):
    torch.set_num_threads(4)
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('decoder')!='coordinates' or data.get('stage')!='finetune': raise ValueError('ต้องใช้ checkpoint รุ่นไม่ใช้ MANO ที่ fine-tune แล้ว')
    net=HandTransformer(**data['config']); net.load_state_dict(data['model']); net.eval()
    return net,data

model,metadata=load_model(str(checkpoint),checkpoint.stat().st_mtime_ns)
records=json.loads((ROOT/'data/manifest.json').read_text())['records']
records=[r for r in records if r['label'] in metadata['labels']]
st.info(f"ชุดทดลอง {len(metadata['labels'])} คำ • ยังไม่มีผลทดสอบกับผู้ทำท่าคนใหม่")
a,b=st.columns([1,1.2])
with a:
    mode=st.radio('เลือกข้อมูลเข้า',['คลิปในชุดฝึก','อัปโหลดคลิปใหม่'],horizontal=True)
    uploaded=None; chosen=None
    if mode=='คลิปในชุดฝึก':
        chosen=st.selectbox('คำตัวอย่าง',records,format_func=lambda r:r['label'])
        st.video(chosen['video'])
        st.caption('การลองคลิปในชุดฝึกแสดงการจดจำตัวอย่าง ไม่ใช่การวัดความแม่นยำกับคนใหม่')
    else:
        uploaded=st.file_uploader('วิดีโอท่าภาษามือ 1 คำ (ไม่เกิน 15 วินาที)',type=['mp4','mov','webm'])
        if uploaded: st.video(uploaded.getvalue())
    run=st.button('ทำนายคำ',type='primary',disabled=chosen is None and uploaded is None)
with b:
    st.subheader('ผลทำนาย 5 อันดับ')
    if not run: st.write('เลือกคลิปแล้วกดทำนายคำ')
    else:
        temporary=None
        try:
            if chosen:
                with np.load(chosen['features'],allow_pickle=False) as d:
                    idx=np.linspace(0,len(d['xy'])-1,metadata['config']['frames']).round().astype(int)
                    x=d['xy'][idx]; valid=d['visible'][idx]; coverage=d['coverage']
            else:
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix,delete=False) as f:
                    f.write(uploaded.getvalue()); temporary=Path(f.name)
                import cv2
                cap=cv2.VideoCapture(str(temporary)); fps=cap.get(cv2.CAP_PROP_FPS); count=cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release()
                if fps<=0 or count<2 or count/fps>15: raise ValueError('ต้องใช้คลิปที่อ่านได้และยาวไม่เกิน 15 วินาที')
                from prepare import features
                from extract_keypoints import extract_video_keypoints
                with st.spinner('กำลังสกัดจุดมือ…'):
                    raw=extract_video_keypoints(str(temporary))
                    x,valid,coverage=features(raw,frames=metadata['config']['frames'])
            with torch.no_grad(): probabilities=model(torch.from_numpy(x)[None],torch.from_numpy(valid)[None]).softmax(-1)[0]
            scores,indices=probabilities.topk(min(5,len(probabilities)))
            rows=[{'อันดับ':i+1,'คำ':metadata['labels'][idx],'คะแนนโมเดล':round(float(score),4)} for i,(score,idx) in enumerate(zip(scores,indices.tolist()))]
            st.dataframe(rows,hide_index=True,use_container_width=True)
            st.caption('คะแนน softmax ไม่ใช่ความมั่นใจที่สอบเทียบแล้ว โมเดลเลือกจากคำในชุดฝึกเสมอ')
            st.write('ตรวจพบจุดมือ: ซ้าย {:.0%} · ขวา {:.0%}'.format(*coverage))
            matching=next((r for r in records if r['label']==rows[0]['คำ']),None)
            if matching: st.video(matching['video'])
        except Exception as exc: st.error(f'ประมวลผลไม่สำเร็จ: {exc}')
        finally:
            if temporary: temporary.unlink(missing_ok=True)
