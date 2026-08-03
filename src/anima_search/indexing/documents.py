from anima_search.schemas import ImageAnnotation


def annotation_to_document(annotation: ImageAnnotation) -> str:
    fields = [("摘要", [annotation.summary]), ("主体", annotation.objects),
              ("动作", annotation.actions), ("场景", [annotation.scene]),
              ("属性", annotation.attributes), ("关系", annotation.spatial_relations),
              ("风格", annotation.style), ("情绪", annotation.mood),
              ("颜色", annotation.colors), ("文字", annotation.ocr_text)]
    return " ".join(f"{name}:{' '.join(values)}" for name, values in fields if any(values))
