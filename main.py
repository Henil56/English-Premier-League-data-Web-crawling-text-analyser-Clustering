"""
===============================================================================
EPL Insight Engine — Main Pipeline Orchestrator
===============================================================================
Entry point for the complete analysis pipeline:

    Stage 1: Web Crawling     — Fetch articles from BBC, Sky, EPL
    Stage 2: Preprocessing    — Clean and tokenize text
    Stage 3: Text Analysis    — TF-IDF, word frequency, sentiment
    Stage 4: Clustering       — K-Means with auto optimal K
    Stage 5: Visualization    — Charts, word clouds, scatter plots
    Stage 6: Report           — Console summary + CSV exports

Usage:
    python main.py              # Full pipeline (demo data)
    python main.py --live       # Full pipeline (live crawling)
    python main.py --demo       # Force demo data mode

Author  : EPL Insight Engine Team
Course  : Web Crawling + Text Analysis + Clustering
Domain  : English Premier League
Sources : BBC Sport, Sky Sports, premierleague.com
===============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

# Project modules
from config import (
    BASE_DIR, OUTPUT_DIR, CRAWLED_DATA_DIR,
    ANALYSIS_RESULTS_DIR, VISUALIZATIONS_DIR,
    USE_DEMO_DATA, CLUSTER_CONFIG,
)
from crawlers import BBCCrawler, SkyCrawler, EPLCrawler
from analysis.preprocessor import TextPreprocessor
from analysis.text_analyzer import TextAnalyzer
from analysis.clustering import ClusteringEngine
from visualization.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EPL-Pipeline")


# =============================================================================
# DEMO DATA — Realistic EPL articles for demonstration
# =============================================================================
def generate_demo_data():
    """
    Generate realistic sample EPL articles for demonstration.

    Returns 60 articles across 5 topic categories:
        - Transfer News (signings, deals, rumours)
        - Match Reports (results, scorelines, performances)
        - Injury Updates (fitness, recovery, absences)
        - Managerial News (appointments, sackings, tactics)
        - Tactical Analysis (formations, statistics, strategy)

    Each article has: title, content, source, date, category.

    Returns:
        list: List of article dicts.
    """
    logger.info("Generating demo dataset (60 EPL articles)...")

    articles = [
        # ===== TRANSFER NEWS (12 articles) =====
        {
            "title": "Arsenal Complete £65m Deal for Top Striker",
            "content": "Arsenal have completed the signing of a prolific striker in a deal worth sixty five million pounds. The forward arrives from Bundesliga and has scored over twenty goals in each of his last three seasons. Manager Mikel Arteta expressed his delight at securing the signature of a player who will add significant firepower to the squad. The transfer represents Arsenal largest ever outlay on a single player and demonstrates the clubs ambition to compete for the title. The striker is expected to partner with Bukayo Saka and Martin Odegaard in the forward line. Sporting director Edu described the deal as a statement of intent from the club. The player has signed a five year contract and will wear the number nine shirt.",
            "source": "BBC Sport", "date": "2025-07-15", "category": "Transfer News"
        },
        {
            "title": "Chelsea Target Midfield Reinforcement with £45m Bid",
            "content": "Chelsea have submitted a forty five million pound bid for a highly rated central midfielder from Serie A. The twenty three year old has been impressive in Italian football and is viewed as the perfect addition to bolster the midfield. Chelsea are looking to add creativity and drive to their engine room after a mixed season. Manager Enzo Maresca has identified the midfielder as his top priority for the window. The player is known for his passing range and ability to control the tempo of matches. Negotiations are ongoing and Chelsea remain confident of completing the deal before the transfer deadline. The midfielder has also attracted interest from Bayern Munich and Barcelona.",
            "source": "Sky Sports", "date": "2025-07-18", "category": "Transfer News"
        },
        {
            "title": "Manchester United Announce Shock Signing from La Liga",
            "content": "Manchester United have announced the surprise signing of a versatile defender from La Liga in a deal worth thirty eight million pounds. The defender can play across the back line and brings significant experience from European competition. United boss Erik ten Hag praised the players intelligence and tactical flexibility. The signing addresses a key weakness in the squad and adds depth to a defence that struggled with injuries last season. The player underwent a medical at Carrington before signing a four year deal. United fans have responded enthusiastically on social media with the announcement video receiving over ten million views in the first hour.",
            "source": "PremierLeague.com", "date": "2025-07-20", "category": "Transfer News"
        },
        {
            "title": "Liverpool Close to Agreeing Fee for Japanese International",
            "content": "Liverpool are on the verge of completing a transfer for a highly coveted Japanese international winger. The deal is expected to be worth around fifty million pounds with add-ons potentially taking the total to sixty million. The winger has been outstanding in Ligue Un scoring fourteen goals and providing eleven assists last season. Arne Slot sees the player as the ideal replacement for Mohamed Salah whose contract situation remains unresolved. The Japanese international is known for his blistering pace and ability to cut inside from the right flank. Liverpool have beaten competition from several top European clubs to secure his services.",
            "source": "BBC Sport", "date": "2025-08-01", "category": "Transfer News"
        },
        {
            "title": "Tottenham Agree Record Fee for Argentine Playmaker",
            "content": "Tottenham Hotspur have agreed a club record fee of fifty five million pounds for an Argentine playmaker. The creative midfielder has been a standout performer and his ability to unlock defences with incisive passing makes him an exciting addition. Ange Postecoglou has been keen to add more creativity to his attacking system and views the Argentine as the missing piece. The player is expected to operate as the number ten behind the striker and could form a devastating partnership with Son Heung-min. The deal includes performance-related bonuses that could increase the overall package significantly.",
            "source": "Sky Sports", "date": "2025-08-05", "category": "Transfer News"
        },
        {
            "title": "Newcastle Break Transfer Record with £70m Forward Swoop",
            "content": "Newcastle United have shattered their transfer record by signing a prolific forward for seventy million pounds. The signing represents a major statement from the Saudi-backed ownership and signals the clubs ambition to establish themselves among the elite. The forward has experience playing in the Champions League and scored consistently at the highest level. Eddie Howe described the acquisition as transformative and believes the player will elevate the entire squad. Newcastle fans gathered outside St James Park to welcome the new signing who was paraded on the pitch before a packed stadium.",
            "source": "PremierLeague.com", "date": "2025-08-10", "category": "Transfer News"
        },
        {
            "title": "West Ham Complete Triple Swoop in £80m Spending Spree",
            "content": "West Ham United have completed a triple signing worth a combined eighty million pounds as they look to rebuild under their new manager. The three arrivals include a goalkeeper from Portugal a centre back from the Netherlands and an attacking midfielder from Brazil. The club has moved decisively in the transfer market after a disappointing campaign last season. The signings demonstrate the boards commitment to giving the new manager the resources needed to succeed. Each player has signed a long term contract and will be integrated into the squad during pre-season training.",
            "source": "BBC Sport", "date": "2025-07-22", "category": "Transfer News"
        },
        {
            "title": "Aston Villa Land Promising Wonderkid from Ajax Academy",
            "content": "Aston Villa have signed a highly talented eighteen year old wonderkid from the Ajax academy in a deal worth fifteen million pounds. The young midfielder has already made twenty appearances in the Eredivisie and represented the Netherlands at under twenty one level. Villa boss Unai Emery is known for developing young talent and sees enormous potential in the teenager. The signing continues Villas strategy of investing in emerging talent alongside established players. The wonderkid will initially train with the first team squad before being gradually introduced to Premier League football.",
            "source": "Sky Sports", "date": "2025-07-25", "category": "Transfer News"
        },
        {
            "title": "Everton Seal Loan Deal for Experienced Defender",
            "content": "Everton have completed a season-long loan deal for an experienced centre back from Juventus. The thirty year old defender brings a wealth of experience having won multiple domestic titles and played extensively in the Champions League. Everton needed reinforcements at the back after selling their primary centre back earlier in the window. The loan deal includes an option to buy at the end of the season which could be triggered based on appearances and performances. The defender is expected to provide leadership and stability to a young Everton backline.",
            "source": "PremierLeague.com", "date": "2025-08-15", "category": "Transfer News"
        },
        {
            "title": "Brighton Reject Third Bid for Star Midfielder",
            "content": "Brighton and Hove Albion have rejected a third bid from Manchester City for their star midfielder. The latest offer of forty million pounds was deemed insufficient by the Brighton board who value the player at over sixty million. The midfielder has been outstanding for Brighton scoring eight goals and providing twelve assists from midfield. Brighton are determined to keep their prized asset and have insisted the player is not for sale. Manager Roberto De Zerbi has built his tactical system around the midfielder and losing him would significantly impact the squad. The transfer saga is expected to continue until the deadline.",
            "source": "BBC Sport", "date": "2025-08-08", "category": "Transfer News"
        },
        {
            "title": "Crystal Palace Sign Exciting Winger from Championship",
            "content": "Crystal Palace have signed a rapid winger from the Championship in a deal worth twelve million pounds. The twenty one year old impressed last season with his direct running and ability to create chances. Palace see the signing as one for the future with the winger expected to develop under the guidance of more experienced teammates. The player scored nine goals and provided seven assists in the Championship last season attracting attention from several Premier League clubs. Palace moved quickly to secure the deal and the winger has expressed his excitement at playing at the highest level.",
            "source": "Sky Sports", "date": "2025-07-28", "category": "Transfer News"
        },
        {
            "title": "Wolves Complete Surprise Deadline Day Signing",
            "content": "Wolverhampton Wanderers have pulled off a surprise deadline day signing by securing a Portuguese international from Benfica for twenty five million pounds. The attacking midfielder adds flair and creativity to the Wolves squad and his Portuguese connections at the club should help him settle quickly. The deal was completed just hours before the transfer window closed after intensive last-minute negotiations. Wolves fans were thrilled by the late addition to the squad which addresses a key area of weakness in the team.",
            "source": "PremierLeague.com", "date": "2025-09-01", "category": "Transfer News"
        },

        # ===== MATCH REPORTS (12 articles) =====
        {
            "title": "Arsenal Thrash Liverpool 4-1 in Statement Victory",
            "content": "Arsenal produced a stunning performance to demolish Liverpool four goals to one at the Emirates Stadium in what could prove to be a defining result in the title race. Bukayo Saka was the star of the show scoring twice and providing an assist in a masterful display. Arsenal took the lead after just eight minutes when Saka curled a beautiful shot into the far corner. Liverpool equalised through Mohamed Salah but Arsenal responded emphatically with three goals in a devastating fifteen minute spell in the second half. The result moves Arsenal to the top of the table and sends a clear message to their rivals.",
            "source": "BBC Sport", "date": "2025-10-05", "category": "Match Report"
        },
        {
            "title": "Manchester City Edge Past Chelsea in Five Goal Thriller",
            "content": "Manchester City edged past Chelsea three goals to two in a breathtaking encounter at the Etihad Stadium amid the ongoing backdrop of 114 charges brought against the club by the Premier League. Erling Haaland scored twice including a last minute winner to seal all three points for the defending champions who continue to perform despite facing 114 charges related to financial fair play breaches. Chelsea had fought back from two goals down to level the score and looked set to earn a point before Haaland struck in stoppage time. The match was a showcase of attacking football with both teams creating numerous chances. Pep Guardiola praised his teams resilience while acknowledging that Chelsea had pushed them to the limit. The result keeps City in the title hunt as they look to defend their crown while the 114 charges hearing continues in the background.",
            "source": "Sky Sports", "date": "2025-10-06", "category": "Match Report"
        },
        {
            "title": "Stunning Comeback at Old Trafford as United Rescue Point",
            "content": "Manchester United produced a remarkable second half comeback to draw three all with Tottenham at Old Trafford. United trailed three nil at half time after a torrid opening forty five minutes but staged an incredible fightback. Bruno Fernandes inspired the revival with a goal and two assists as United scored three times in the final twenty minutes. The comeback will go down as one of the great Old Trafford fightbacks. Manager Erik ten Hag praised the character and fighting spirit shown by his players but acknowledged that the first half performance was unacceptable. The result extends Uniteds unbeaten run at home but questions remain about their inconsistency.",
            "source": "PremierLeague.com", "date": "2025-10-19", "category": "Match Report"
        },
        {
            "title": "Newcastle Dominate Aston Villa with Clinical Display",
            "content": "Newcastle United put on a clinical display to beat Aston Villa three nil at St James Park. Alexander Isak scored a brace either side of a Bruno Guimaraes goal as Newcastle controlled proceedings from start to finish. Villa had no answer to Newcastles intensity and pressing which restricted the visitors to just one shot on target. Eddie Howe was delighted with the performance describing it as the most complete display of the season. The victory continues Newcastles excellent home form and moves them into the top four. Villa manager Unai Emery admitted his side were second best in every department.",
            "source": "BBC Sport", "date": "2025-11-02", "category": "Match Report"
        },
        {
            "title": "Brighton Shock Manchester City with Tactical Masterclass",
            "content": "Brighton produced a tactical masterclass to stun Manchester City two nil at the Amex Stadium as the cloud of 114 charges continues to hang over the champions. The result dealt a significant blow to Citys title hopes and demonstrated Brightons growing quality under their innovative manager. Brighton pressed City relentlessly and exploited the spaces left by the champions high defensive line. Two goals in quick succession midway through the second half sealed a famous victory. The result was Citys first defeat in twelve league matches and the 114 charges saga adds further uncertainty to the clubs future. Brighton players celebrated wildly at full time as the home crowd created an electric atmosphere.",
            "source": "Sky Sports", "date": "2025-11-09", "category": "Match Report"
        },
        {
            "title": "Liverpool Cruises Past Everton in Merseyside Derby",
            "content": "Liverpool cruised to a comfortable three nil victory over Everton in the Merseyside derby at Anfield. Arne Slots side were dominant throughout with goals from Virgil van Dijk Cody Gakpo and Diogo Jota sealing the win. Everton rarely threatened and were outclassed in every area of the pitch. Liverpool controlled possession and created chance after chance against a beleaguered Everton defence. The result piles more pressure on the Everton manager whose position is increasingly under scrutiny. Liverpool fans enjoyed a celebratory atmosphere at Anfield as their team moved to within two points of the league leaders.",
            "source": "PremierLeague.com", "date": "2025-12-07", "category": "Match Report"
        },
        {
            "title": "West Ham Hold Manchester United to Goalless Draw",
            "content": "West Ham held Manchester United to a goalless draw at the London Stadium in a match that failed to live up to expectations. Both sides created chances but lacked the cutting edge to find a breakthrough. United dominated possession but were frustrated by a well-organised West Ham defence that restricted them to long range efforts. The draw is a point gained for West Ham in their fight against relegation but represents two points dropped for United in their pursuit of a top four finish. Both managers expressed frustration at the result with each believing their side should have won the match.",
            "source": "BBC Sport", "date": "2025-12-14", "category": "Match Report"
        },
        {
            "title": "Wolves Stun Tottenham with Last-Minute Winner",
            "content": "Wolves stunned Tottenham Hotspur with a dramatic last minute winner to claim a two one victory at Molineux. The result was a devastating blow for Spurs who had fought hard to equalise after falling behind in the first half. Wolves defended resolutely throughout the second half before breaking away in the final minute to score through their substitute striker. The goal sparked wild celebrations among the home fans and left the Tottenham players devastated. The result moves Wolves up to fifteenth in the table and provides a much needed boost in their battle to avoid relegation.",
            "source": "Sky Sports", "date": "2025-12-21", "category": "Match Report"
        },
        {
            "title": "Crystal Palace Earn Vital Point Against Arsenal",
            "content": "Crystal Palace earned a hard fought one one draw against Arsenal at Selhurst Park in a match that could have significant implications for the title race. Palace took the lead through a stunning long range strike that flew into the top corner leaving the Arsenal goalkeeper with no chance. Arsenal pressed relentlessly for an equaliser and were rewarded when their substitute headed home from a corner in the seventy eighth minute. Palace can feel proud of their performance against one of the title favourites and the point could prove vital in their battle to stay in the division.",
            "source": "PremierLeague.com", "date": "2026-01-11", "category": "Match Report"
        },
        {
            "title": "Bournemouth Record Historic Victory Over Chelsea",
            "content": "Bournemouth recorded a historic four nil victory over Chelsea at the Vitality Stadium in one of the most remarkable results of the season. The Cherries were unstoppable from the first whistle with their pressing and counter attacking causing Chelsea constant problems. Three goals in the first half put the game beyond Chelsea before a fourth was added in injury time. It was Bournemouths largest ever victory over Chelsea and a result that will live long in the memory of their supporters. The manager praised his players for executing the game plan perfectly and producing a performance of the highest quality.",
            "source": "BBC Sport", "date": "2026-01-25", "category": "Match Report"
        },
        {
            "title": "Dramatic North London Derby Ends in Four-Goal Thriller",
            "content": "The north London derby between Arsenal and Tottenham ended in a thrilling two two draw at the Emirates Stadium. The match lived up to its billing as one of the most intense rivalries in English football with both teams producing moments of brilliance. Arsenal twice took the lead but Tottenham showed tremendous character to equalise on both occasions. The atmosphere inside the Emirates was electric with both sets of supporters creating a cacophony of noise throughout the ninety minutes. The result keeps Arsenal top of the table but Tottenham will feel they could have snatched all three points after their strong finish.",
            "source": "Sky Sports", "date": "2026-02-08", "category": "Match Report"
        },
        {
            "title": "Leicester City Complete Great Escape with Final Day Win",
            "content": "Leicester City completed a remarkable great escape from relegation with a dramatic two one victory on the final day of the season. Needing to win to guarantee their survival Leicester fell behind early and looked destined for the drop. However two goals in the final fifteen minutes turned the match on its head and sparked scenes of unbridled joy. The comeback victory ensures Leicester will play Premier League football next season and represents one of the most dramatic survival stories in recent memory. Players and fans celebrated together long after the final whistle in emotional scenes.",
            "source": "PremierLeague.com", "date": "2026-05-25", "category": "Match Report"
        },

        # ===== INJURY UPDATES (12 articles) =====
        {
            "title": "Key Arsenal Striker Ruled Out for Six Weeks with Hamstring Injury",
            "content": "Arsenal have been dealt a significant blow with confirmation that their key striker will miss the next six weeks due to a hamstring injury sustained during training. The forward pulled up during a routine sprint exercise and scans have revealed a grade two hamstring tear. The injury comes at the worst possible time for Arsenal who face a crucial run of fixtures including matches against Manchester City and Chelsea. Manager Mikel Arteta confirmed the news at his press conference and admitted it was a big loss for the squad. The medical team will work closely with the player to ensure a full recovery and avoid any setback that could extend the absence.",
            "source": "BBC Sport", "date": "2025-09-20", "category": "Injury Update"
        },
        {
            "title": "Liverpool Captain Faces Extended Spell on Sidelines",
            "content": "Liverpool captain Virgil van Dijk is facing an extended spell on the sidelines after suffering a knee injury during the Champions League match. Initial assessments suggest the central defender could be out for up to three months. The injury is a devastating blow for Liverpool whose defensive record has been significantly better with van Dijk in the team. Medical staff are exploring all treatment options and the player will see a specialist later this week. Arne Slot acknowledged the severity of the situation and confirmed that the club will not rush the recovery process. Liverpool have options at centre back but losing their captain and leader is a significant setback.",
            "source": "Sky Sports", "date": "2025-10-15", "category": "Injury Update"
        },
        {
            "title": "Chelsea Midfielder Returns to Training After Long Injury Layoff",
            "content": "Chelsea midfielder has returned to full training after a lengthy injury layoff that kept him sidelined for four months. The player suffered ankle ligament damage during a tackle in August and has been working tirelessly with the rehabilitation team to regain fitness. His return is a massive boost for Chelsea who have missed his dynamism and energy in the middle of the park. The manager confirmed that the midfielder will be eased back into action gradually and could feature as a substitute in the upcoming fixture. The players dedication to his recovery has been praised by coaches and teammates alike who are delighted to see him back on the training pitch.",
            "source": "PremierLeague.com", "date": "2025-12-01", "category": "Injury Update"
        },
        {
            "title": "Manchester City Hit by Defensive Injury Crisis",
            "content": "Manchester City are facing a defensive injury crisis with three senior centre backs ruled out of this weekends crucial match adding to the pressure from the 114 charges they face. The injuries have left Pep Guardiola with limited options at the back and he may be forced to deploy a midfielder in defence. The defensive problems are compounded by the loss of the right back who also picked up a knock in training. City have been particularly unfortunate with injuries this season with the treatment room rarely empty amid the distraction of 114 charges. Guardiola expressed his frustration at the situation but insisted his squad has the quality and depth to cope despite the charges. The manager is expected to make tactical adjustments to compensate for the missing defenders while the club continues to deal with the 114 charges hearing.",
            "source": "BBC Sport", "date": "2025-11-28", "category": "Injury Update"
        },
        {
            "title": "Tottenham Star Winger Undergoes Successful Surgery",
            "content": "Tottenham Hotspur have confirmed that their star winger has undergone successful surgery on a fractured metatarsal. The player is expected to be sidelined for approximately ten to twelve weeks during the recovery period. The injury was sustained during a challenge in the match against Wolverhampton Wanderers and the winger was seen leaving the stadium on crutches. Spurs will miss his creativity and goal threat during a demanding period of fixtures. Ange Postecoglou confirmed that the surgery went well and the rehabilitation programme will begin immediately. The club is hopeful the player will return before the end of the season.",
            "source": "Sky Sports", "date": "2025-11-05", "category": "Injury Update"
        },
        {
            "title": "Newcastle Goalkeeper Sidelined with Shoulder Problem",
            "content": "Newcastle United goalkeeper has been ruled out for three weeks with a shoulder injury sustained during a collision in the penalty area. The keeper landed awkwardly while attempting to punch clear a corner and immediately signalled to the bench that he was in pain. Scans have confirmed damage to the rotator cuff which will require rest and physiotherapy treatment. The backup goalkeeper will deputise during the absence and has experience of starting in the Premier League. Manager Eddie Howe expressed confidence in his replacement and emphasised the importance of the goalkeeper position being covered by capable understudies.",
            "source": "PremierLeague.com", "date": "2025-10-22", "category": "Injury Update"
        },
        {
            "title": "Aston Villa Receive Boost as Two Key Players Return",
            "content": "Aston Villa have received a timely boost with the return to fitness of two key players who have been missing for several weeks. The defensive midfielder and left back have both been declared fit to play after recovering from separate muscle injuries. Their return comes at a crucial time for Villa who have struggled in their absence losing three of their last four matches. Manager Unai Emery confirmed both players trained fully with the squad and are available for selection. The return of the defensive midfielder is particularly significant as Villa have conceded more goals without him in the team.",
            "source": "BBC Sport", "date": "2026-01-15", "category": "Injury Update"
        },
        {
            "title": "Everton Forward Suffers Recurrence of Groin Injury",
            "content": "Everton forward has suffered a recurrence of the groin injury that kept him out for six weeks earlier in the season. The striker felt discomfort during the warm up before the match and was subsequently withdrawn from the squad. The recurrence is a worrying development for both the player and the club who were hoping the issue had been fully resolved. Everton will seek further medical advice to determine the best course of treatment and the forward may require a longer recovery period this time. The injury leaves Everton short of attacking options during a critical phase of the season.",
            "source": "Sky Sports", "date": "2026-02-05", "category": "Injury Update"
        },
        {
            "title": "Manchester United Midfielder Picks Up International Injury",
            "content": "Manchester United are sweating on the fitness of their midfielder who picked up an injury while on international duty. The player limped off during the second half of his national team match with what appeared to be a calf strain. United are awaiting a full assessment from the medical staff but initial reports suggest the injury is not as serious as first feared. The midfielder has been in excellent form recently and his absence would be keenly felt. International injuries are a constant source of frustration for club managers who have no control over how their players are managed during the break.",
            "source": "PremierLeague.com", "date": "2025-11-18", "category": "Injury Update"
        },
        {
            "title": "Brighton Confirm ACL Injury for Young Defender",
            "content": "Brighton have confirmed that their promising young defender has suffered an anterior cruciate ligament injury that will keep him out for the remainder of the season. The twenty year old sustained the injury during a training ground collision and was immediately taken for scans which confirmed the worst fears. The ACL injury is a devastating blow for the young player who had been establishing himself as a regular starter. Brighton manager expressed his sympathy and assured the player he would receive the best possible medical care. The club will provide full support during what is expected to be a nine to twelve month recovery period.",
            "source": "BBC Sport", "date": "2026-03-10", "category": "Injury Update"
        },
        {
            "title": "Chelsea Dealt Triple Injury Blow Ahead of Cup Final",
            "content": "Chelsea have been dealt a triple injury blow ahead of their cup final with three key players ruled out of the showpiece event. The captain goalkeeper and first choice left back have all failed fitness tests and will not be available for selection. The injuries leave manager Enzo Maresca with difficult decisions to make as he prepares for the biggest match of the season. Chelsea will need to reshuffle their lineup and rely on squad players to step up for the occasion. The absences are a significant disadvantage against an opponent who will have their full squad available.",
            "source": "Sky Sports", "date": "2026-04-20", "category": "Injury Update"
        },
        {
            "title": "Premier League Injury Report: Most Affected Clubs This Season",
            "content": "A comprehensive analysis of injury data across the Premier League season reveals significant disparities between clubs in terms of player availability. The most affected club has lost over two hundred and fifty player match days to injury compared to fewer than one hundred for the least affected. Soft tissue injuries account for the majority of absences with hamstring problems being the most common type. The data highlights the physical demands of the modern game and the challenges clubs face in managing player workload. Sports scientists have called for greater investment in injury prevention programmes and more intelligent fixture scheduling.",
            "source": "PremierLeague.com", "date": "2026-05-10", "category": "Injury Update"
        },

        # ===== MANAGERIAL NEWS (12 articles) =====
        {
            "title": "Everton Sack Manager After Poor Start to Season",
            "content": "Everton have sacked their manager following a dismal start to the season that has left the club in the relegation zone. The decision was taken after a run of seven defeats in ten matches which has left the club fighting for survival. The board issued a statement thanking the manager for his efforts but acknowledging that a change was necessary to give the club the best chance of staying in the Premier League. An interim manager will take charge while the club conducts a thorough search for a permanent replacement. The sacking continues a pattern of managerial instability at Everton who are now looking for their fourth manager in three years.",
            "source": "BBC Sport", "date": "2025-11-10", "category": "Managerial News"
        },
        {
            "title": "Former International Manager Takes Over at Struggling Club",
            "content": "A former international team manager has been appointed as the new head coach of a struggling Premier League club. The experienced manager brings a wealth of knowledge from both domestic and international football and is tasked with steering the club away from relegation danger. The appointment has been broadly welcomed by fans who see the new manager as a safe pair of hands during a difficult period. The manager held his first press conference and outlined his plans to improve the teams defensive record which has been the worst in the division. He emphasised the importance of organisation discipline and hard work as the foundation for improvement.",
            "source": "Sky Sports", "date": "2025-11-15", "category": "Managerial News"
        },
        {
            "title": "Pep Guardiola Signs Contract Extension at Manchester City",
            "content": "Manchester City have announced that Pep Guardiola has signed a contract extension keeping him at the club until the end of next season despite the ongoing 114 charges investigation. The Spaniard has been in charge since arriving and has transformed the club into the dominant force in English football. Under his leadership City have won multiple Premier League titles and a Champions League trophy despite facing 114 charges from the Premier League for alleged financial breaches. Guardiola expressed his continued hunger for success and his desire to build a lasting legacy at the club. The 114 charges have cast a shadow over the clubs achievements but Guardiola insists the squad remains focused. The announcement ends months of speculation about the managers future and provides stability during a period overshadowed by the 114 charges hearing. City fans celebrated the news with the hashtag StayPep trending on social media.",
            "source": "PremierLeague.com", "date": "2025-11-20", "category": "Managerial News"
        },
        {
            "title": "Arteta Praises Team Spirit After Crucial Victory",
            "content": "Arsenal manager Mikel Arteta praised the incredible team spirit and mentality shown by his players after a crucial victory that moved the club top of the Premier League table. Arteta highlighted the collective effort and unity within the squad as key factors in their success. The manager has built a strong team culture since arriving at the club and the results are evident in the players performances on the pitch. Arteta also credited his coaching staff for their meticulous preparation and attention to detail. The manager emphasised that while individual quality is important the strength of the team lies in their togetherness and shared belief in the project.",
            "source": "BBC Sport", "date": "2025-12-20", "category": "Managerial News"
        },
        {
            "title": "Postecoglou Addresses Questions Over Tactical Approach",
            "content": "Tottenham manager Ange Postecoglou has addressed questions over his tactical approach following criticism from some fans and pundits. The Australian manager has been committed to an attacking style of play since taking charge but recent results have raised questions about whether the approach is sustainable. Postecoglou defended his philosophy and insisted that the long term vision is clear even if short term results have been inconsistent. He argued that building an identity and playing style takes time and patience. The manager pointed to the improved performances in recent weeks as evidence that the players are adapting to his methods. Postecoglou received backing from the club board who continue to support his vision.",
            "source": "Sky Sports", "date": "2026-01-08", "category": "Managerial News"
        },
        {
            "title": "New Manager Bounce: Club Wins Three in a Row After Appointment",
            "content": "The newly appointed manager has overseen three consecutive victories since taking charge of the struggling club in what pundits are calling a classic new manager bounce. The results have lifted the team out of the relegation zone and provided a much needed confidence boost for the players. The manager has made subtle tactical changes and improved the teams defensive organisation without sacrificing attacking intent. Players have spoken positively about the new regime and the training methods which they describe as demanding but enjoyable. The challenge now is to maintain this momentum and turn the initial improvement into sustained progress throughout the remainder of the season.",
            "source": "PremierLeague.com", "date": "2025-12-10", "category": "Managerial News"
        },
        {
            "title": "Slot Implements New Pressing System at Liverpool",
            "content": "Liverpool manager Arne Slot has implemented a new pressing system that has transformed the teams performance in recent matches. The Dutch manager arrived with a reputation for tactical innovation and has introduced a more structured approach to pressing compared to his predecessors heavy metal style. The new system involves coordinated triggers for pressing and clear instructions on when to hold position. Players have reported that the tactical clarity has improved their confidence and understanding of their roles. The statistics support the eye test with Liverpool recording the highest number of ball recoveries in the attacking third this season. Slot believes the system will yield even better results as the players become more familiar with the nuances.",
            "source": "BBC Sport", "date": "2026-02-15", "category": "Managerial News"
        },
        {
            "title": "Championship Manager Linked with Premier League Job",
            "content": "A highly rated Championship manager has emerged as the leading candidate for a vacant Premier League position after impressing with his innovative coaching methods. The forty two year old has led his current club to the top of the second tier with an attractive brand of football that has caught the attention of scouts and recruitment teams across the top flight. The manager is known for his detailed tactical preparation data driven approach and ability to develop young players. Negotiations are reportedly at an advanced stage with the Premier League club willing to pay significant compensation for his services. The appointment would represent a step up in level but many believe the manager is ready for the challenge.",
            "source": "Sky Sports", "date": "2026-03-05", "category": "Managerial News"
        },
        {
            "title": "Manager of the Month Award Goes to Surprise Candidate",
            "content": "The Premier League Manager of the Month award has gone to a surprise candidate who guided his team to four wins from four matches during a remarkable unbeaten run. The managers tactical acumen and ability to motivate his players have been key factors in the clubs resurgence. The award recognises the outstanding work done during a month in which the team exceeded all expectations. The manager dedicated the award to his coaching staff and players and credited the support of the fans for creating an environment in which the team could thrive. The unexpected recognition highlights the competitive nature of the Premier League where any team can produce exceptional results on their day.",
            "source": "PremierLeague.com", "date": "2026-03-15", "category": "Managerial News"
        },
        {
            "title": "Howe Discusses Newcastle's European Ambitions",
            "content": "Newcastle United manager Eddie Howe has spoken candidly about the clubs European ambitions and their desire to establish themselves as regular competitors in continental competition. Howe acknowledged that the journey from mid table to European qualification requires sustained investment in both the squad and infrastructure. The manager outlined his vision for building a team capable of competing on multiple fronts without sacrificing domestic performance. Howe praised the club ownership for their long term strategic planning and patient approach to building a competitive squad. He emphasised that while the ambition is clear the process must be gradual and sustainable to ensure lasting success.",
            "source": "BBC Sport", "date": "2026-04-01", "category": "Managerial News"
        },
        {
            "title": "Relegation Battle Intensifies as Two More Managers Face Pressure",
            "content": "The relegation battle has intensified with two more managers coming under increasing pressure following a series of poor results. Both clubs sit in the bottom three with time running out to avoid the drop. The boards of both clubs are understood to be considering their options with potential replacements already being sounded out. The precarious position of the managers reflects the brutal nature of the Premier League where results are the ultimate currency. Neither manager attended the post match press conference after their most recent defeats fuelling speculation about their futures. The next round of fixtures could prove decisive in determining whether either manager survives.",
            "source": "Sky Sports", "date": "2026-04-15", "category": "Managerial News"
        },
        {
            "title": "End of Season Review: Best and Worst Managerial Appointments",
            "content": "As the Premier League season draws to a close a review of the managerial appointments made during the campaign reveals a mixed picture of success and failure. The best appointment brought stability and improved results taking a struggling club from the relegation zone to mid table safety. The worst appointment resulted in a further decline in performance and an eventual sacking after just eight matches in charge. The analysis highlights the importance of cultural fit tactical compatibility and recruitment support in determining whether a managerial appointment succeeds. The turnover of managers this season has been among the highest in Premier League history reflecting the increasing demands and diminishing patience at the top level of English football.",
            "source": "PremierLeague.com", "date": "2026-05-20", "category": "Managerial News"
        },

        # ===== TACTICAL ANALYSIS (12 articles) =====
        {
            "title": "How Arsenal's New Formation Changed the Title Race",
            "content": "Arsenal tactical switch to a three at the back formation has been a game changer in the title race. The new system has provided greater defensive stability while maintaining the attacking threat that has been the hallmark of Artetas side. The formation allows the wing backs to provide width and the extra centre back gives more security against counter attacks. Data analysis shows that Arsenal have conceded significantly fewer goals since the tactical change while maintaining their expected goals output in attack. The switch was prompted by a series of defensive errors that cost Arsenal points early in the season. Artetas willingness to adapt and evolve his tactical approach has been praised by analysts who credit the manager for his flexibility and strategic intelligence.",
            "source": "BBC Sport", "date": "2026-02-20", "category": "Tactical Analysis"
        },
        {
            "title": "Pressing Stats Reveal Most Intense Teams in the League",
            "content": "A detailed statistical analysis of pressing intensity across the Premier League has revealed the most and least aggressive teams in terms of pressing. The data measures the number of pressures applied per opponent possession and the success rate of those pressures. The results show a clear correlation between pressing intensity and league position with the top four teams all ranking highly for pressing metrics. The analysis also reveals that effective pressing is not just about effort but about coordination and timing. Teams that press in a disorganised manner tend to leave gaps that opponents can exploit. The most effective pressing teams combine high intensity with tactical discipline creating a suffocating environment for opponents.",
            "source": "Sky Sports", "date": "2026-03-01", "category": "Tactical Analysis"
        },
        {
            "title": "Set Piece Revolution: How Dead Balls Are Deciding Matches",
            "content": "Set pieces have become an increasingly decisive factor in Premier League matches with a growing number of goals being scored from dead ball situations. The trend reflects a broader tactical evolution as teams invest more resources in set piece coaching and preparation. Specialist set piece coaches have become commonplace at Premier League clubs with dedicated training sessions designed to maximise the threat from corners free kicks and throw ins. The data shows that goals from set pieces have increased by twenty percent compared to last season. Teams that score frequently from set pieces tend to have a significant advantage in tight matches where open play goals are hard to come by. The set piece revolution represents a shift in tactical thinking that is reshaping the way matches are won and lost.",
            "source": "PremierLeague.com", "date": "2026-03-20", "category": "Tactical Analysis"
        },
        {
            "title": "The Rise of the Inverted Full-Back in Modern Football",
            "content": "The inverted full back role has become one of the most discussed tactical innovations in modern football with several Premier League managers adopting the approach. The concept involves full backs moving inside into midfield positions rather than staying wide in traditional overlapping runs. Pep Guardiola is widely credited with popularising the role at Manchester City where it has been a key feature of their tactical system. The inverted full back helps control the central midfield area and provides additional passing options when building from the back. The role requires players with excellent technical ability and tactical intelligence. Several other Premier League clubs have begun implementing similar approaches with varying degrees of success.",
            "source": "BBC Sport", "date": "2026-04-05", "category": "Tactical Analysis"
        },
        {
            "title": "Expected Goals Analysis: The Most Clinical and Wasteful Teams",
            "content": "An expected goals analysis of the Premier League season reveals which teams have been the most clinical and most wasteful in front of goal. Expected goals measures the quality of chances created based on historical data from similar opportunities. The most clinical team has outperformed their expected goals by a significant margin scoring considerably more goals than the statistical model predicted. Conversely the most wasteful team has underperformed their expected goals significantly suggesting that their finishing quality has been below par. The analysis provides valuable insights into team performance that go beyond simple goals scored and conceded statistics. Coaches and analysts use expected goals data to identify areas for improvement and to assess whether current performance levels are sustainable.",
            "source": "Sky Sports", "date": "2026-04-10", "category": "Tactical Analysis"
        },
        {
            "title": "Defensive Masterclass: How Low Block Teams Frustrate the Big Six",
            "content": "Several Premier League clubs have adopted a low defensive block strategy that has proven remarkably effective against the league top teams. The low block involves dropping deep and compact with ten outfield players behind the ball to deny space for the opposition to play through. The strategy requires immense discipline concentration and physical fitness as defending for extended periods is mentally and physically demanding. Teams employing the low block have recorded impressive results against top six opposition this season with multiple upsets caused by this approach. The tactical approach relies on quick counter attacks to exploit the spaces left by teams pushing forward. Critics argue the low block is negative and anti football but managers who employ it counter that it is a legitimate and effective tactical approach that maximises their teams strengths.",
            "source": "PremierLeague.com", "date": "2026-04-25", "category": "Tactical Analysis"
        },
        {
            "title": "Data Deep Dive: Pass Networks Reveal Team Playing Styles",
            "content": "Pass network analysis has emerged as a powerful tool for understanding and comparing team playing styles in the Premier League. By mapping the frequency and direction of passes between players analysts can visualise how teams build attacks and identify key passing connections. The analysis reveals distinct playing styles among Premier League clubs from possession based approaches with intricate short passing patterns to more direct styles with fewer passes and greater vertical progression. The pass networks also highlight the importance of specific players within a teams tactical framework. Players who consistently feature as central nodes in the pass network are often the most influential in terms of dictating the tempo and direction of play.",
            "source": "BBC Sport", "date": "2026-05-01", "category": "Tactical Analysis"
        },
        {
            "title": "Counter-Pressing: Why Winning the Ball Back Immediately Matters",
            "content": "Counter pressing also known as gegenpressing has become a fundamental tactical concept in the Premier League with several teams employing the strategy to great effect. The approach involves immediately pressing the opposition the moment possession is lost rather than retreating into a defensive shape. The logic behind counter pressing is that the opposition is most vulnerable to losing the ball in the moments immediately after winning it when players are out of their defensive positions. Teams that counter press effectively can win the ball back in dangerous areas and create high quality scoring opportunities. The statistical data shows that goals scored within five seconds of a turnover have increased significantly reflecting the growing emphasis on counter pressing in modern tactical planning.",
            "source": "Sky Sports", "date": "2026-05-05", "category": "Tactical Analysis"
        },
        {
            "title": "Wing Play vs Central Overloads: Contrasting Attacking Approaches",
            "content": "A tactical comparison of Premier League attacking approaches reveals two dominant philosophies: traditional wing play and central overloads. Teams that favour wing play utilise wide players to stretch the opposition defence and deliver crosses into the penalty area. This approach relies on pace and technical quality from wingers and a target forward who excels in aerial duels. In contrast teams that create central overloads pack the middle of the pitch with technical players who combine through short passes and movements to penetrate the defence. The choice between these approaches often depends on the personnel available and the opponents defensive structure. Some of the most successful teams have the ability to switch between both approaches within a match making them unpredictable and difficult to defend against.",
            "source": "PremierLeague.com", "date": "2026-05-15", "category": "Tactical Analysis"
        },
        {
            "title": "Goalkeeper Distribution: The New Battleground in Tactical Warfare",
            "content": "Goalkeeper distribution has become a crucial tactical consideration in the Premier League with shot stoppers now expected to contribute significantly to their teams build up play. Modern goalkeepers are assessed not just on their ability to make saves but on their passing accuracy under pressure and their ability to initiate attacks from the back. The best distributing goalkeepers can bypass the opponents press with accurate long passes or build patiently from the back with short passes to the centre backs. Statistical analysis shows a correlation between effective goalkeeper distribution and overall team performance suggesting that this aspect of the game is becoming increasingly important. Several clubs have specifically targeted goalkeepers with strong distribution skills in the transfer market recognising the tactical advantage they provide.",
            "source": "BBC Sport", "date": "2026-05-18", "category": "Tactical Analysis"
        },
        {
            "title": "Transition Speed: How Quickly Teams Switch Between Attack and Defence",
            "content": "Transition speed the time it takes for a team to switch from defensive to attacking mode and vice versa has become a key performance indicator in the Premier League. Analysis of tracking data reveals significant differences between teams in their transition speed with the fastest teams completing the switch in under four seconds. Quick transitions allow teams to exploit moments of defensive disorganisation and create high quality scoring opportunities before the opposition can establish their defensive shape. Conversely teams that are slow in their defensive transitions are more vulnerable to counter attacks. Coaches now dedicate specific training sessions to improving transition speed recognising its importance in determining match outcomes.",
            "source": "Sky Sports", "date": "2026-05-22", "category": "Tactical Analysis"
        },
        {
            "title": "Season Finale: Tactical Trends That Defined the Campaign",
            "content": "As the Premier League season reaches its conclusion several tactical trends have emerged that have defined the campaign and are likely to influence future seasons. The increased use of data analytics in tactical decision making has been one of the most significant developments with clubs investing heavily in data science departments. Other notable trends include the evolution of the false nine role the growing importance of set piece coaching the rise of inverted full backs and the emphasis on counter pressing. The tactical sophistication of the Premier League continues to increase with managers constantly seeking innovative ways to gain competitive advantages. The trend towards more structured and organised defensive systems has also been evident with low blocks becoming increasingly common against stronger opposition. These tactical trends reflect the broader evolution of football as a sport that is becoming ever more analytical and strategically complex.",
            "source": "PremierLeague.com", "date": "2026-05-28", "category": "Tactical Analysis"
        },
    ]

    logger.info(f"Generated {len(articles)} demo articles across 5 categories")

    # Add missing fields
    for article in articles:
        article["crawled_at"] = datetime.now().isoformat()
        article["url"] = f"https://example.com/{article['title'].lower().replace(' ', '-')[:50]}"

    return articles


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def create_output_dirs():
    """Create all output directories."""
    for dir_path in [OUTPUT_DIR, CRAWLED_DATA_DIR, ANALYSIS_RESULTS_DIR, VISUALIZATIONS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Output directories ready: {OUTPUT_DIR}")


def run_pipeline(use_demo=True):
    """
    Execute the complete EPL Insight Engine pipeline.

    Pipeline Stages:
        1. Web Crawling or Demo Data Generation
        2. Text Preprocessing (NLTK)
        3. Text Analysis (TF-IDF, Sentiment, Word Frequency)
        4. K-Means Clustering with Optimal K Detection
        5. Visualization Generation
        6. Results Export and Console Report

    Args:
        use_demo (bool): If True, use demo data; if False, crawl live.
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print("🏟️  EPL INSIGHT ENGINE — Analysis Pipeline")
    print("=" * 70)
    print(f"   Mode    : {'📋 Demo Data' if use_demo else '🌐 Live Crawling'}")
    print(f"   Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Create output directories ---
    create_output_dirs()

    # =========================================================================
    # STAGE 1: DATA COLLECTION
    # =========================================================================
    print("\n" + "─" * 70)
    print("📡 STAGE 1: Data Collection")
    print("─" * 70)

    if use_demo:
        all_articles = generate_demo_data()
    else:
        all_articles = []

        # Instantiate crawlers
        crawlers = [
            BBCCrawler(),
            SkyCrawler(),
            EPLCrawler(),
        ]

        # Run each crawler
        for crawler in crawlers:
            try:
                articles = crawler.run()
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Crawler {crawler.source_name} failed: {e}")

        # Fallback to demo if no articles were crawled
        if not all_articles:
            logger.warning("No articles crawled. Falling back to demo data.")
            all_articles = generate_demo_data()

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    print(f"\n   ✅ Total articles collected: {len(df)}")
    print(f"   📊 Sources: {df['source'].value_counts().to_dict()}")
    print(f"   📂 Categories: {df['category'].value_counts().to_dict()}")

    # Save raw data
    raw_path = os.path.join(CRAWLED_DATA_DIR, "all_articles.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
    print(f"   💾 Raw data saved: {raw_path}")

    # =========================================================================
    # STAGE 2: TEXT PREPROCESSING
    # =========================================================================
    print("\n" + "─" * 70)
    print("🔧 STAGE 2: Text Preprocessing")
    print("─" * 70)

    preprocessor = TextPreprocessor()

    # Preprocess article content
    df["cleaned_content"] = preprocessor.preprocess_batch(df["content"].tolist())

    # Also preprocess titles
    df["cleaned_title"] = preprocessor.preprocess_batch(df["title"].tolist())

    # Combine title + content for analysis
    df["full_text"] = df["cleaned_title"] + " " + df["cleaned_content"]

    # Stats
    non_empty = df["full_text"].str.strip().astype(bool).sum()
    avg_len = df["full_text"].str.split().str.len().mean()
    print(f"\n   ✅ Preprocessed {len(df)} documents")
    print(f"   📝 Non-empty documents: {non_empty}")
    print(f"   📏 Avg document length: {avg_len:.0f} words")

    # =========================================================================
    # STAGE 3: TEXT ANALYSIS
    # =========================================================================
    print("\n" + "─" * 70)
    print("📊 STAGE 3: Text Analysis (TF-IDF + Sentiment + Word Frequency)")
    print("─" * 70)

    analyzer = TextAnalyzer()

    # 3a. TF-IDF
    tfidf_matrix = analyzer.compute_tfidf(df["full_text"].tolist())
    top_terms = analyzer.get_top_tfidf_terms(n=20)
    print(f"\n   ✅ TF-IDF: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

    # 3b. Word Frequency
    word_freqs = analyzer.compute_word_frequencies(df["full_text"].tolist())
    print(f"   ✅ Words: {word_freqs['total_words']:,} total, {word_freqs['unique_words']:,} unique")

    # 3c. Per-source word frequency
    source_freqs = analyzer.compute_word_frequencies_by_source(df, text_column="full_text")

    # 3d. Sentiment Analysis
    df = analyzer.analyze_sentiment(df, text_column="content")
    print(f"   ✅ Sentiment: {df['sentiment_label'].value_counts().to_dict()}")

    # =========================================================================
    # STAGE 4: CLUSTERING
    # =========================================================================
    print("\n" + "─" * 70)
    print("🔬 STAGE 4: K-Means Clustering")
    print("─" * 70)

    clustering = ClusteringEngine()

    # 4a. Find optimal K
    optimal_k = clustering.find_optimal_k(tfidf_matrix)
    print(f"\n   ✅ Optimal K: {optimal_k}")

    # 4b. Fit K-Means
    labels = clustering.fit(tfidf_matrix, k=optimal_k)
    df["cluster"] = labels

    # 4c. Extract keywords per cluster
    keywords = clustering.get_cluster_keywords(analyzer.tfidf_vectorizer)

    # 4d. PCA for visualization
    pca_coords = clustering.reduce_dimensions(tfidf_matrix)

    # Print cluster info
    print(f"\n   📊 Cluster distribution:")
    for cluster_id in range(optimal_k):
        count = (labels == cluster_id).sum()
        top_kw = ", ".join([kw[0] for kw in keywords[cluster_id][:5]])
        print(f"      Cluster {cluster_id} ({count:2d} articles): {top_kw}")

    # =========================================================================
    # STAGE 5: VISUALIZATION
    # =========================================================================
    print("\n" + "─" * 70)
    print("📈 STAGE 5: Generating Visualizations")
    print("─" * 70)

    viz = Visualizer()
    saved_plots = viz.generate_all(df, df["full_text"].tolist(), clustering, analyzer)
    print(f"\n   ✅ Generated {len(saved_plots)} visualizations")
    for name, path in saved_plots.items():
        print(f"      📊 {name}: {os.path.basename(path)}")

    # =========================================================================
    # STAGE 6: SAVE RESULTS & REPORT
    # =========================================================================
    print("\n" + "─" * 70)
    print("💾 STAGE 6: Saving Results")
    print("─" * 70)

    # Save analysis results
    analyzer.save_results(df)
    clustering.save_results(df)

    # Save complete DataFrame
    final_path = os.path.join(ANALYSIS_RESULTS_DIR, "complete_analysis.csv")
    df.to_csv(final_path, index=False, encoding="utf-8")
    print(f"   ✅ Complete analysis saved: {final_path}")

    # Print summaries
    analyzer.print_summary()
    clustering.print_summary()

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("🏆  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   ⏱️  Total time    : {elapsed:.1f} seconds")
    print(f"   📄  Articles      : {len(df)}")
    print(f"   🔤  Vocabulary    : {len(analyzer.feature_names):,} terms")
    print(f"   🔬  Clusters      : {optimal_k}")
    print(f"   📊  Visualizations: {len(saved_plots)}")
    print(f"\n   📂  Output folder : {OUTPUT_DIR}")
    print(f"       ├── crawled_data/       → Raw articles (JSON)")
    print(f"       ├── analysis_results/   → CSV exports")
    print(f"       └── visualizations/     → Charts & plots (PNG)")
    print("=" * 70 + "\n")

    return df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="EPL Insight Engine — Web Crawling + Text Analysis + Clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run with demo data (default)
    python main.py --live       # Run with live web crawling
    python main.py --demo       # Force demo data mode
        """,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live web crawling instead of demo data",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Force demo data mode (default)",
    )

    args = parser.parse_args()

    # Determine mode
    use_demo = True  # Default
    if args.live:
        use_demo = False
    elif args.demo:
        use_demo = True
    else:
        use_demo = USE_DEMO_DATA

    # Run pipeline
    try:
        run_pipeline(use_demo=use_demo)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
