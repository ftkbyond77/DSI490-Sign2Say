from pathlib import Path
import tempfile
import streamlit as st
from core import references, label, extract, distance

st.set_page_config(page_title='ภาษามือ • Demo',page_icon='🤟',layout='wide')
st.title('ค้นหาคำจากภาษามือ')
st.caption('Demo ทดลองจับคู่คลิปกับคำอ้างอิง 5 คำ • ประมวลผลบนเครื่องนี้')
try:
    refs=references()
except ValueError as exc:
    st.error(str(exc)); st.stop()

@st.cache_data(show_spinner=False)
def database(stamps):
    return [extract(p) for p,_,_ in stamps]

left,right=st.columns([1,1.2],gap='large')
with left:
    mode=st.radio('เลือกคลิปที่ต้องการค้นหา',['ใช้คลิปตัวอย่าง','อัปโหลดคลิป'],horizontal=True)
    payload=None
    if mode=='ใช้คลิปตัวอย่าง':
        source=st.selectbox('คลิปตัวอย่าง',refs,format_func=label)
        st.video(str(source))
        st.caption('การเลือกคลิปเดียวกับฐานข้อมูลเป็นการตรวจการทำงาน ยังไม่ใช่การวัดความแม่นยำกับคนใหม่')
    else:
        source=None
        upload=st.file_uploader('คลิปท่าภาษามือ 1 คำ',type=['mp4','mov','avi','webm'])
        st.caption('ใช้คลิปสั้นไม่เกิน 15 วินาที เห็นศีรษะ ไหล่ และมือทั้งสองข้าง')
        if upload:
            payload=upload.getvalue(); st.video(payload)
    run=st.button('ค้นหาคำใกล้เคียง',type='primary',disabled=source is None and payload is None)
with right:
    if not run:
        st.subheader('คำอ้างอิงใน demo')
        for p in refs: st.write('• '+label(p))
        st.info('เลือกคลิปแล้วกดค้นหา การประมวลผลครั้งแรกใช้เวลาสักครู่')
    else:
        temporary=None
        try:
            if payload is not None:
                with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix,delete=False) as f:
                    f.write(payload); temporary=Path(f.name)
                source=temporary
            import cv2
            cap=cv2.VideoCapture(str(source))
            fps=cap.get(cv2.CAP_PROP_FPS); frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps<=0 or frames<=0: raise ValueError('อ่านข้อมูลวิดีโอไม่ได้ กรุณาใช้ไฟล์ MP4 ที่เล่นได้')
            if frames/fps>15: raise ValueError('กรุณาตัดคลิปให้ยาวไม่เกิน 15 วินาที')
            with st.spinner('กำลังอ่านท่าภาษามือและเปรียบเทียบคลิป…'):
                db=database(tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in refs))
                query=extract(source)
                ranked=sorted(zip(refs,db),key=lambda pair:distance(query,pair[1]))
            st.subheader('ผลใกล้เคียง 5 อันดับ')
            st.caption('ระยะ DTW ยิ่งต่ำยิ่งคล้าย • ระบบจะจัดอันดับเสมอ แม้คำที่ทำจะไม่อยู่ในชุดอ้างอิง')
            st.dataframe([{'อันดับ':i+1,'คำ':label(p),'ระยะ DTW':round(distance(query,d),4)} for i,(p,d) in enumerate(ranked)],hide_index=True,use_container_width=True)
            st.video(str(ranked[0][0]))
            st.caption('คลิปอ้างอิงอันดับ 1: '+label(ranked[0][0]))
            st.write('สัดส่วนเฟรมที่ตรวจพบ: ไหล่ {:.0%} · มือซ้าย {:.0%} · มือขวา {:.0%}'.format(float(query['pose_coverage']),float(query['left_coverage']),float(query['right_coverage'])))
            if min(float(query['left_coverage']),float(query['right_coverage']))<0.5:
                st.warning('ตรวจพบรายละเอียดมือในบางเฟรมเท่านั้น ผลอาจอาศัยการเคลื่อนไหวแขนเป็นหลัก')
        except Exception as exc:
            st.error(f'ประมวลผลไม่สำเร็จ: {exc}')
        finally:
            if temporary: temporary.unlink(missing_ok=True)
