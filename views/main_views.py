from flask import Blueprint, render_template, url_for
from werkzeug.utils import redirect
from app.models import Getrecommender

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/hello')
def hello_recommender():
    return 'Hello, Recommender!'

@bp.route('/')
def index():
    return redirect(url_for('compare.input'))