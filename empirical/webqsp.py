kmap=r'm.068kst common.topic.notable_types m.01m9 | m.02qdj5n people.person.place_of_birth m.05k_nc | m.04k9r1 people.person.religion m.0631_ | m.01bw1c book.author.works_written m.06t_tbs | m.0268jbf people.person.place_of_birth m.059rby | m.01l6v_ location.location.contains m.010pp7c5 | m.02wntpq education.education.institution m.07tk7 | m.02m7r people.person.education m.02kq21s | m.02vl9_c common.topic.notable_types m.01xs05r | m.02vm87b common.topic.notable_types m.01nf | m.026tzb topic_server.population_number "1572" | m.02m7r people.person.education m.02wntpq | m.02m7r people.person.education m.0n1fbly | m.04n11yr common.topic.notable_types m.01y2jks | m.01_rph people.person.profession m.0dxtg | m.02kq21s education.education.institution m.07tg4 | m.026g2jd people.person.nationality m.09c7w0 | m.02m7r people.person.education m.0kv984f | m.01t840 location.location.containedby m.02jx1 | m.09zxwr government.politician.government_positions_held m.0n4qxvr | m.04j1_st media_common.netflix_title.netflix_genres m.0c3351 | m.0bmclg people.person.gender m.05zppz | m.0k3kzn people.person.place_of_birth m.04f_d | m.0h_73rf sports.pro_athlete.teams m.0wch476 | m.0237d4 location.location.contains m.0j_4hqg | m.0hyxv location.location.containedby m.07ssc | m.01z_51 location.location.contains m.03lw32 | m.02k1np location.location.containedby m.0d060g | m.02kqsd location.location.containedby m.02jx1 | m.02kq20k education.education.institution m.07xpm | m.0ct2nx location.location.time_zones m.02lcqs | m.07wk1 location.location.containedby m.0fz25 | m.0cf442 common.topic.notable_types m.04lw | m.065y__d common.topic.notable_types m.01n7 | m.06w1qty location.location.containedby m.014ds6 | m.06yp_y people.deceased_person.place_of_burial m.0lbp_ | m.02rd7y location.location.containedby m.05ksh | m.06w7f48 location.location.containedby m.0d060g | m.0447f people.person.profession m.04gc2 | m.0n1fbly education.education.institution m.02722n'

rules=kmap.split(' | ')
import networkx as nx
import matplotlib.pyplot as plt

g = nx.DiGraph()
for rule in rules:
    u,t,v=rule.split()
    g.add_edge(u, v, name=t)
pos = nx.spring_layout(g, iterations=20)
nx.draw(g,pos,with_labels=True)
nx.draw_networkx_edge_labels(g,pos,font_size=7,edge_labels=nx.get_edge_attributes(g, 'name'))
# plt.savefig('test.png')
plt.show()
print(len(rules))