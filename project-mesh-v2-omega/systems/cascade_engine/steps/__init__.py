"""Cascade step modules — each step in the content cascade pipeline."""

from .base import BaseStep
from .article_step import ArticleStep
from .image_step import ImageStep
from .wordpress_step import WordPressStep
from .video_step import VideoStep
from .social_step import SocialStep
from .product_step import ProductStep
from .internal_link_step import InternalLinkStep
from .email_step import EmailStep

STEP_MAP = {
    "article": ArticleStep,
    "image": ImageStep,
    "wordpress": WordPressStep,
    "video": VideoStep,
    "social": SocialStep,
    "product": ProductStep,
    "internal_link": InternalLinkStep,
    "email": EmailStep,
}

__all__ = ["STEP_MAP", "BaseStep"]
