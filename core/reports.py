from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def build_pdf_report(result):
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    styles=getSampleStyleSheet()
    story=[Paragraph("SmartStandards AI — Compliance & Recommendation Report",styles["Title"])]
    story.append(Spacer(1,12))
    story.append(Paragraph("Extracted requirements",styles["Heading2"]))
    req=result["requirements"]
    for k,v in req.items():
        if k!="raw": story.append(Paragraph(f"<b>{k.title()}:</b> {', '.join(v) if isinstance(v,list) else v}",styles["BodyText"]))
    story.append(Spacer(1,12))
    story.append(Paragraph("Recommended standards",styles["Heading2"]))
    data=[["Standard","Score","Status","Classification"]]
    for r in result["recommendations"]:
        data.append([r["standard_id"],str(r["score"]),r["status"],r["classification"]])
    table=Table(data,colWidths=[110,55,70,100])
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.4,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
    story.append(table)
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Compliance coverage: {result['compliance']['coverage']}%",styles["Heading2"]))
    for x in result["compliance"]["items"]:
        story.append(Paragraph(f"{x['requirement']}: {x['status']}",styles["BodyText"]))
    story.append(Spacer(1,12))
    story.append(Paragraph("Draft procurement specification",styles["Heading2"]))
    for line in result["generated_specification"].splitlines():
        story.append(Paragraph(line.replace("&","&amp;"),styles["BodyText"]))
    doc.build(story)
    return buf.getvalue()

def build_csv_report(result):
    rows=[]
    for r in result["recommendations"]:
        rows.append({"Standard":r["standard_id"],"Title":r["title"],"Score":r["score"],"Status":r["status"],"Classification":r["classification"],"Evidence":"; ".join(r["evidence"])})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
