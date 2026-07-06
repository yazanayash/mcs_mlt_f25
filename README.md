# Handwritten Digit Recognition Web Application

تطبيق ويب للتعرف على الأرقام المكتوبة يدوياً باستخدام Machine Learning.

## الأعضاء
- محمد كريم الخياط — mhd_kareem_341562
- ...

## التقنيات المستخدمة
- Python, TensorFlow/Keras
- Flask / FastAPI
- HTML/CSS/JavaScript
- GitHub

## طريقة التشغيل محلياً
```bash
pip install -r requirements.txt
```

تدريب النموذج (الجزء الجاهز حالياً):
```bash
python -m ml.train
```

الواجهة الخلفية (backend) والواجهة الأمامية (frontend) قيد التطوير.
راجع `CONTRACT.md` لطريقة استخدام النموذج من جهة الـ backend.

## روابط النشر
- التطبيق: ...
- المستودع: ...

## بنية المستودع
- `ml/` : كود الـ Machine Learning (تحميل البيانات، النموذج، التدريب، التقييم، المعالجة المسبقة)
- `notebooks/` : دفاتر تدريب النموذج
- `models/` : النموذج المدرَّب المحفوظ
- `report/` : ملف التسليم النهائي
- `CONTRACT.md` : واجهة الاتفاق بين جهة الـ ML وجهة الـ backend
- `app/` : تطبيق الويب (backend) — لم يُنشأ بعد
- `frontend/` : واجهة المستخدم — لم تُنشأ بعد
