"""Native desktop UI. Worker threads never touch Tk widgets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .detector import ROOT


class App:
    def __init__(self, root):
        self.root = root
        root.title("Детектор мусора · YOLO11")
        root.geometry("1040x760")
        root.minsize(800, 600)
        self.profile = tk.StringVar(value="land")
        self.mode = tk.StringVar(value="image")
        self.source = tk.StringVar()
        self.output = tk.StringVar(value=str(ROOT / "outputs/app"))
        self.conf = tk.StringVar(value="")
        self.operating_mode=tk.StringVar(value='Баланс')
        self.quality=tk.StringVar(value='Обычная')
        self.status = tk.StringVar(value="Выберите профиль и файл. Камера не подключается автоматически.")
        self.events = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.worker = None
        self.closing = False
        top = ttk.Frame(root, padding=18); top.pack(fill="both", expand=True)
        ttk.Label(top, text="Поиск мусора", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(top, text="Фото, папки и видео · обычные и подводные сцены").pack(anchor="w", pady=(0,16))
        options = ttk.Frame(top); options.pack(fill="x")
        ttk.Label(options, text="Профиль:").pack(side="left")
        ttk.Combobox(options,textvariable=self.profile,values=["land","underwater"],state="readonly",width=16).pack(side="left",padx=8)
        ttk.Label(options,text="Режим:").pack(side="left")
        ttk.Combobox(options,textvariable=self.mode,values=["image","folder","video","camera"],state="readonly",width=12).pack(side="left",padx=8)
        ttk.Label(options,text="Confidence (пусто = профиль):").pack(side="left")
        ttk.Entry(options,textvariable=self.conf,width=7).pack(side="left",padx=8)
        quality_row=ttk.Frame(top);quality_row.pack(fill='x',pady=7)
        ttk.Label(quality_row,text='Приоритет:').pack(side='left')
        ttk.Combobox(quality_row,textvariable=self.operating_mode,values=['Баланс','Меньше пропусков','Меньше ложных рамок'],state='readonly',width=23).pack(side='left',padx=8)
        ttk.Label(quality_row,text='Обработка:').pack(side='left')
        ttk.Combobox(quality_row,textvariable=self.quality,values=['Обычная','По фрагментам (SAHI)'],state='readonly',width=23).pack(side='left',padx=8)
        ttk.Label(quality_row,text='По фрагментам: медленнее, для крупных фото').pack(side='left')
        self.path_row(top,"Источник",self.source,self.browse_source)
        self.path_row(top,"Результаты",self.output,self.browse_output)
        buttons=ttk.Frame(top);buttons.pack(fill="x",pady=10)
        self.start_button=ttk.Button(buttons,text="Начать обработку",command=self.start);self.start_button.pack(side="left")
        self.stop_button=ttk.Button(buttons,text="Остановить",command=self.cancel,state="disabled");self.stop_button.pack(side="left",padx=8)
        ttk.Button(buttons,text="Открыть результаты",command=self.open_output).pack(side="right")
        self.progress=ttk.Progressbar(top,mode="indeterminate");self.progress.pack(fill="x")
        ttk.Label(top,textvariable=self.status,wraplength=950).pack(anchor="w",pady=8)
        self.preview=ttk.Label(top,anchor="center",text="Здесь появится обработанное изображение")
        self.preview.pack(fill="both",expand=True)
        ttk.Label(top,text="Подводный профиль работает с файлами. Физические камеры в текущем этапе не проверялись.",wraplength=950).pack(anchor="w",pady=(10,0))
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.after(100,self.poll)

    def path_row(self,parent,label,variable,command):
        row=ttk.Frame(parent);row.pack(fill="x",pady=6)
        ttk.Label(row,text=label,width=12).pack(side="left")
        ttk.Entry(row,textvariable=variable).pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Выбрать…",command=command).pack(side="left",padx=(8,0))

    def browse_source(self):
        if self.mode.get()=="camera":
            self.source.set("0");return
        value=filedialog.askdirectory() if self.mode.get()=="folder" else filedialog.askopenfilename()
        if value:self.source.set(value)

    def browse_output(self):
        value=filedialog.askdirectory()
        if value:self.output.set(value)

    def open_output(self):
        path=Path(self.output.get());path.mkdir(parents=True,exist_ok=True);os.startfile(path)

    def emit(self,event):
        try:self.events.put_nowait(event)
        except queue.Full:
            try:self.events.get_nowait()
            except queue.Empty:pass
            self.events.put_nowait(event)

    def start(self):
        if self.worker and self.worker.is_alive():return
        try:
            confidence=float(self.conf.get()) if self.conf.get().strip() else None
            if confidence is not None and not 0<confidence<=1:raise ValueError()
        except ValueError:
            messagebox.showerror("Настройки","Confidence должен быть от 0 до 1.");return
        operating={'Баланс':'balanced','Меньше пропусков':'recall','Меньше ложных рамок':'precision'}[self.operating_mode.get()]
        args=(self.profile.get(),self.mode.get(),self.source.get(),self.output.get(),confidence,operating,self.quality.get()!='Обычная')
        if not args[2]:messagebox.showerror("Источник","Выберите файл или папку.");return
        if args[1]=="camera":
            if args[-1]:messagebox.showerror('Режим','SAHI предназначен для файлов.');return
            if args[0]=="underwater":
                messagebox.showerror("Профиль","Подводный профиль поддерживает только файлы.");return
            if not messagebox.askyesno("Открыть веб-камеру?","Это откроет обычную физическую веб-камеру. Продолжить?"):return
        self.stop.clear();self.start_button.configure(state="disabled");self.stop_button.configure(state="normal")
        self.progress.start();self.status.set("Загрузка модели…")
        self.worker=threading.Thread(target=self.work,args=args,daemon=True);self.worker.start()

    def work(self,profile,mode,source,output,confidence,operating_mode,sliced):
        try:
            from .detector import Detector,read_image
            from .media import process_folder,process_video
            detector=Detector(profile,confidence,operating_mode=operating_mode,sliced=sliced)
            def preview(image,record):
                self.emit(("preview",image,f"Кадр {record['frame_id']} · объектов: {len(record['detections'])} · {record['inference_ms']:.0f} мс"))
            if mode=="image":
                target,record=detector.image(source,output);preview(read_image(target),record)
            elif mode=="folder":
                process_folder(detector,source,output,self.stop,lambda path,record:preview(read_image(path),record))
            elif mode=="video":
                process_video(detector,source,Path(output)/"detected.mp4",self.stop,preview)
            else:
                from .camera import run_camera
                run_camera(detector,int(source),stop=self.stop,callback=preview)
            # Wait briefly so the final preview remains visible before completion state.
            self.events.put(("done",None,"Остановлено" if self.stop.is_set() else "Готово. Результаты сохранены."))
        except Exception as error:
            self.events.put(("error",None,str(error)))

    def poll(self):
        from PIL import Image,ImageTk
        import cv2
        try:
            while True:
                kind,image,status=self.events.get_nowait();self.status.set(status)
                if image is not None:
                    picture=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    picture.thumbnail((max(1,self.preview.winfo_width()-12),max(1,self.preview.winfo_height()-12)))
                    self.photo=ImageTk.PhotoImage(picture);self.preview.configure(image=self.photo,text="")
                if kind in ("done","error"):
                    self.progress.stop();self.start_button.configure(state="normal");self.stop_button.configure(state="disabled")
                    if kind=="error" and not self.closing:messagebox.showerror("Ошибка обработки",status)
        except queue.Empty:pass
        if self.closing and (not self.worker or not self.worker.is_alive()):self.root.destroy();return
        self.root.after(100,self.poll)

    def cancel(self):
        self.stop.set();self.status.set("Завершение текущего кадра…")

    def close(self):
        self.closing=True;self.stop.set();self.status.set("Закрытие…")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke-test",action="store_true");args=parser.parse_args()
    root=tk.Tk()
    if args.smoke_test:root.withdraw()
    app=App(root)
    if args.smoke_test:
        errors=[]
        def callback_error(exc,value,tb):
            errors.append(str(value));__import__('traceback').print_exception(exc,value,tb)
        root.report_callback_exception=callback_error
        samples=list((ROOT/'data/samples/land').glob('*.jpg'))
        if samples:
            from .detector import read_image
            root.after(120,lambda:app.emit(('preview',read_image(samples[0]),'Software preview check')))
        root.after(400,root.destroy);root.mainloop()
        if errors:raise RuntimeError('; '.join(errors))
        if samples and not hasattr(app,'photo'):raise RuntimeError('Preview callback did not complete')
        print(f"UI constructor and event loop passed; preview checked={bool(samples)}; no camera opened")
    else:root.mainloop()


if __name__=="__main__":main()
