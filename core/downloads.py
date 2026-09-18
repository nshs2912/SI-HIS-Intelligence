"""Export utilities for SI-HIS analytical verification packages."""
import hashlib
import io
import re
import pandas as pd

def _clean_text(text):
    text = str(text or "")
    text = re.sub(r"[*_#>]", "", text)
    text = re.sub(r"[^\x00-\x7FÀ-ÿ\n\r\t]", "", text)
    return text

def data_sha256(df):
    return hashlib.sha256(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).hexdigest()

def build_evaluation_excel(df, metadata=None):
    metadata = metadata or {}
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Data_Analisis", index=False)
        meta = pd.DataFrame([{"Parameter": k, "Nilai": v} for k, v in metadata.items()])
        meta.to_excel(writer, sheet_name="Metadata_Verifikasi", index=False)
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "border": 1, "text_wrap": True})
        for ws_name, frame in [("Data_Analisis", df), ("Metadata_Verifikasi", meta)]:
            ws = writer.sheets[ws_name]
            for col_num, name in enumerate(frame.columns):
                ws.write(0, col_num, name, header_fmt)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns)-1, 0))
            if len(frame.columns):
                ws.set_column(0, len(frame.columns)-1, 18)
    out.seek(0)
    return out.getvalue()

def build_evaluation_csv(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def build_resume_pdf(title, resume_text, metadata=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from xml.sax.saxutils import escape
    metadata = metadata or {}
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm, title=title, author="SI-HIS Intelligence")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("SIHISTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=16, leading=20, spaceAfter=8)
    h_style = ParagraphStyle("SIHISH", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("SIHISBody", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=5)
    small = ParagraphStyle("SIHISSmall", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.grey)
    story = [Paragraph(escape(title), title_style)]
    if metadata:
        rows = [["Parameter", "Nilai"]] + [[escape(str(k)), escape(str(v))] for k, v in metadata.items()]
        table = Table(rows, colWidths=[48*mm, 122*mm], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.35,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTSIZE",(0,0),(-1,-1),7.5),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
        story += [table, Spacer(1, 7)]
    for raw in str(resume_text or "").splitlines():
        line = _clean_text(raw).strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), h_style))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), h_style))
        else:
            story.append(Paragraph(escape(line), body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(escape("Dokumen ini merupakan salinan hasil analisis SI-HIS dan datasheet yang digunakan pada saat analisis. Untuk verifikasi, cocokkan SHA-256 datasheet dengan metadata dan gunakan Data_Analisis sebagai sumber baris yang dianalisis. Hasil SI-HIS adalah Decision Support System; konfirmasi dan keputusan operasional tetap pada otoritas kesehatan."), small))
    doc.build(story)
    out.seek(0)
    return out.getvalue()
