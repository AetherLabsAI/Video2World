"""Require a new object-goal transition when a candidate starts already complete.

Reuses the native predicates, including their original AND/OR grouping and
thresholds. Robot home and open-gripper conditions are not object goals. No
minimum motion threshold is introduced. Sequential native tasks retain their
own state machines; this guard only blocks an entirely pre-satisfied task.
"""
VERSION='robodojo-completion-transition/1'
ROBOT_CONDITIONS={'all_robot_back_to_origin','is_robot_back_to_origin','is_robot_not_back_to_origin','is_all_gripper_open'}

def object_only(check):
    if isinstance(check,tuple):
        return None if check[0] in ROBOT_CONDITIONS else check
    if isinstance(check,list):
        kept=[v for x in check if (v:=object_only(x)) is not None]
        return kept or None
    raise ValueError('Unsupported native predicate structure')

class CompletionTransition:
    def __init__(self,reward_manager):
        self.rm=reward_manager
        self.groups=[g for group in reward_manager.check_list[0] if (g:=object_only(group)) is not None]
        if not self.groups:
            raise ValueError('Native task has no inspectable object-goal conditions')
        self.initial_goal=self.goal();self.left_goal=not self.initial_goal
        self.reentered=False;self.frames_observed=1
    def goal(self):
        # A check-list group is AND; nested lists use native alternating OR/AND.
        return all(self.rm.check_once(check,0) for group in self.groups for check in group)
    def observe(self):
        current=self.goal();self.frames_observed+=1
        if not current:self.left_goal=True
        elif self.initial_goal and self.left_goal:self.reentered=True
    @property
    def allowed(self):return not self.initial_goal or self.reentered
    def result(self):
        return dict(protocol=VERSION,initial_object_goal=self.initial_goal,left_object_goal=self.left_goal,reentered_object_goal=self.reentered,completion_allowed=self.allowed,frames_observed=self.frames_observed,reason='ok' if self.allowed else 'initial_object_goal_without_exit_reentry',scope='native object predicate conjunction, with robot-home/gripper-only checks removed; initially complete tasks require exit and re-entry')
