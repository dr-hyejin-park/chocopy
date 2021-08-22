from flask import Blueprint, render_template, request, url_for, g
from ..models import Getrecommender, Useritemmodel, sims_reco, user_history
from ..forms import CompareForm, useridForm
from datetime import datetime
from werkzeug.utils import redirect
from .. import db
from ..utils import get_viewed_list, get_compare_list, get_popular_list

bp = Blueprint('recommend', __name__, url_prefix='/recommend')


@bp.route('/list/')
def _list():
    recommend_list = Getrecommender.query.order_by(Getrecommender.eventTime.desc())
    return render_template('recommend/recommend_list.html', recommend_list=recommend_list)



@bp.route('/detail/')
def detail():
    user_id = g.user.user_id
    reco_set = sims_reco.query.get_or_404(user_id)
    history = user_history.query.get_or_404(user_id)
    reco_list = get_reco_list(user_id)
    popular_list = get_popular_list()
    view_list = get_viewed_list(user_id)
    aws_list = get_aws_list(user_id)
    return render_template('recommend/recommend_detail.html',
                                    reco_set=reco_set,
                                    history=history,
                                    reco_list=reco_list,
                                    popular_list=popular_list,
                                    data_list=view_list,
                                    aws_list =aws_list)
