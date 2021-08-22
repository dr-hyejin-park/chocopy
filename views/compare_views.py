from datetime import datetime

from flask import Blueprint, url_for, request, flash, render_template, session, g
from werkzeug.utils import redirect

from app import db
from app.forms import CompareForm, useridForm
from app.models import Getrecommender, user_history, sims_reco
from ..utils import get_viewed_list, get_popular_list, get_compare_list

bp = Blueprint('compare', __name__, url_prefix='/compare')

@bp.route('/')
@bp.route('/<int:user_id>')
def input(user_id=None):
    popular_list = get_popular_list()
    return render_template('recommend/compare_detail.html', 
                                    user_id=user_id,
                                    popular_list=popular_list)


@bp.errorhandler(404)
def page_not_found(error): 
    return redirect(url_for('compare.input', user_id=None))

@bp.route('/test', methods=['POST'])
def test(user_id=None):
    if request.method == 'POST':
        user = request.form['user_id']
        session.clear()
        session['user_id'] = user
        history = user_history.query.get_or_404(user)
        reco_list, aws_list = get_compare_list(user, 'cf_reco', 'aws_reco')
        view_list = get_viewed_list(user)
        return render_template('recommend/compare_detail.html',
                                user_id=user,
                                history=history,
                                reco_list=reco_list,
                                data_list=view_list,
                                aws_list =aws_list)
    else:
        user = None
    return redirect(url_for('compare.input', user_id=user))


@bp.before_app_request
def load_reco_for_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = sims_reco.query.get(user_id)


@bp.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('compare.input'))



