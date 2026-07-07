# Handwritten Digit Recognition Web Application

تطبيق ويب للتعرف على الأرقام المكتوبة يدوياً باستخدام Machine Learning.

## الأعضاء
- محمد كريم الخياط — mhd_kareem_341562
- علي فائز اسماعيل — ali_351343
- زاهر فايز كامل  — zaher_344165

## التقنيات المستخدمة
- Python, TensorFlow/Keras
- Flask
- HTML/CSS/JavaScript
- GitHub

## طريقة التشغيل محلياً
```bash
pip install -r requirements.txt
```

تدريب النموذج:
```bash
python -m ml.train
```

تشغيل تطبيق الويب:
```bash
python app/main.py
```


## روابط النشر
- التطبيق: ...
- المستودع: ...

## بنية المستودع
- `ml/` : كود الـ Machine Learning (تحميل البيانات، النموذج، التدريب، التقييم، المعالجة المسبقة)
- `notebooks/` : دفاتر تدريب النموذج
- `models/` : النموذج المدرَّب المحفوظ
- `report/` : ملف التسليم النهائي
- `CONTRACT.md` : واجهة الاتفاق بين جهة الـ ML وجهة الـ backend
- `app/` : تطبيق الويب (backend)
- `frontend/` : واجهة المستخدم
